"""RAG pipeline: load docs -> chunk -> embed (local) -> retrieve -> answer."""
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from pypdf import PdfReader

import chat_local

load_dotenv()
log = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
CACHE_FILE = INDEX_DIR / "answer_cache.json"
# Comma-separated; if one is rate-limited (free-tier daily caps) the next is tried.
GEN_MODELS = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-3.7-flash,gemini-3.8-flash").split(",")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6
POOL = 40  # candidates taken from each of vector search and keyword search before fusion
# Scraped pages that are duplicates or placeholder text and only add noise to retrieval.
EXCLUDE_SOURCES = {"test.txt", "message-from-senior-director45435625325342fdsadxcsadxs.txt"}

SYSTEM_PROMPT = """You are the official AI assistant of LIET (Lloyd Institute of Engineering & Technology), Greater Noida.

How to answer:
1. Use the provided LIET website/document excerpts as your primary source. Answer as LIET's own assistant: say "LIET" or "our college" and never describe it as "a private college", "private engineering colleges" or "colleges in Greater Noida" in general.
2. The excerpts include blog articles that use generic phrases like "top private colleges" or mention other institutions. Treat those as being about LIET, and never present other colleges' details as LIET's.
3. If the excerpts do not contain the answer and the question is about LIET specifics (fees, dates, names, phone numbers, policies), do not guess. Say you don't have that detail and give LIET's contact: admissions@liet.in, +91 9821582662.
4. If the question is general (study advice, career, technology, entrance exams, or anything not specific to LIET), answer helpfully from your own knowledge, starting with "Not from LIET's website:". Where it fits, connect the answer back to LIET's relevant programs.
5. Never start a reply with "Sorry" and never apologise. For greetings or small talk (hi, thanks, how are you), reply warmly in the user's language and offer help with LIET topics. If a question is vague, ask one short clarifying question.
6. Be concise and friendly. Do not mention file names, sources, page numbers or the word "context/excerpts" in your answer; the app shows sources separately.

Hard questions:
9. For multi-part questions, answer each part the excerpts cover and say plainly which part the website does not state. Never fill a gap with guessed numbers, percentages, company names, rules or dates.
10. If the excerpts give different figures for the same thing (fees, placements, eligibility), show each figure with the session/year it belongs to and tell the student to confirm the current one with admissions. Do not claim one page is more official unless a page says so.
11. For cost or comparison questions, use only published figures, show the arithmetic, list the charges the website does not publish, and mark the total as an estimate to be confirmed.
12. Rules set by the affiliating university (AKTU), AICTE or the government (exam rules, degree award, attendance norms, scholarships) are not LIET's own. Say so and point to the official document or the admissions contact.
13. Questions about what the website "hides", contradicts or advertises vs. regulations cannot be answered from it. Say that, then list what a student should verify and with whom.

Language (very important):
6a. Detect the language of the user's latest question and reply in the same one:
   - English question -> English answer.
   - Hindi written in Devanagari (e.g. "फीस कितनी है?") -> Hindi in Devanagari, in simple everyday words.
   - Hinglish, i.e. Hindi written in Roman letters (e.g. "hostel ki fees kitni hai?") -> Hinglish in Roman letters, the way a friendly Indian student would write it. Do not switch to Devanagari.
   If the user explicitly asks for a language, use that one.
7. The excerpts are in English. Translate facts into the reply language, but keep names, course names (B.Tech, MBA), numbers, amounts (Rs / INR), emails and phone numbers exactly as they are.
8. Translate the "Not from LIET's website:" prefix and the "I don't have that detail" message into the reply language too."""


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=key)


def load_documents() -> list[dict]:
    """Return [{'source', 'page', 'text'}] for every .txt/.md/.pdf in docs/."""
    pages = []
    for path in sorted(DOCS_DIR.rglob("*")):
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            pages.append({"source": path.name, "page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")})
        elif suffix == ".pdf":
            for i, page in enumerate(PdfReader(path).pages, start=1):
                pages.append({"source": path.name, "page": i, "text": page.extract_text() or ""})
    return [p for p in pages if p["text"].strip()]


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i:i + CHUNK_SIZE].strip()]


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)  # runs locally: no API quota


def _embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    model = _embedder()
    gen = model.query_embed(texts) if is_query else model.embed(texts, batch_size=64)
    arr = np.array(list(gen), dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def build_index() -> int:
    """Embed everything in docs/ and save to index/. Returns number of chunks."""
    chunks = []
    for page in load_documents():
        for piece in chunk_text(page["text"]):
            chunks.append({"source": page["source"], "page": page["page"], "text": piece})
    if not chunks:
        raise RuntimeError(f"No readable documents found in {DOCS_DIR}/")
    vectors = _embed([c["text"] for c in chunks])
    INDEX_DIR.mkdir(exist_ok=True)
    CACHE_FILE.unlink(missing_ok=True)  # cached answers are stale once the knowledge base changes
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return len(chunks)


def index_exists() -> bool:
    return (INDEX_DIR / "vectors.npy").exists() and (INDEX_DIR / "chunks.json").exists()


@lru_cache(maxsize=1)
def _load_index(mtime: float) -> tuple[np.ndarray, list[dict]]:
    return (
        np.load(INDEX_DIR / "vectors.npy"),
        json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8")),
    )


def _needs_rewrite(question: str) -> bool:
    """True when the question is Hindi/Hinglish, which the English-only embedder cannot match well."""
    return chat_local.detect_language(question) != "en"


def _to_english(question: str, history: list[dict] | None) -> str:
    """Rewrite a Hindi/Hinglish question as a standalone English search query. Falls back to the original."""
    convo = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in (history or [])[-2:])
    prompt = (
        "Rewrite the user's latest question as one standalone English search query for a college "
        "website (LIET, Greater Noida). The question may be Hindi (Devanagari) or Hinglish (Hindi in "
        "Roman letters). Resolve pronouns using the conversation. Output only the query.\n\n"
        f"Conversation:\n{convo}\nQuestion: {question}"
    )
    config = types.GenerateContentConfig(temperature=0)
    for model in GEN_MODELS:
        if _dead_until.get(model, 0) > time.time():
            continue
        try:
            text = (_client().models.generate_content(model=model, contents=prompt, config=config).text or "").strip()
            return text or question
        except Exception as e:  # quota, overload, bad key/model name: fall back to keyword expansion
            _dead_until[model] = time.time() + (3600 if getattr(e, "code", None) == 429 else 30)
            log.warning("query rewrite failed on %s: %s", model, e)
    return chat_local.expand_query(question)


_STOP = set("a an the of in on at to for is are was were be been do does did can could i my me we you your it its and or if "
            "what which who whom how when where why with by from that this as any there than then so not no".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@lru_cache(maxsize=1)
def _bm25_index(mtime: float) -> tuple[dict[str, list[tuple[int, int]]], np.ndarray, float]:
    """Inverted index {term: [(chunk_id, tf)]}, chunk lengths and their mean, for BM25 keyword search."""
    _, chunks = _load_index(mtime)
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths = np.zeros(len(chunks), dtype=np.float32)
    for i, c in enumerate(chunks):
        toks = _tokens(c["text"])
        lengths[i] = len(toks)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            postings.setdefault(t, []).append((i, n))
    return postings, lengths, float(lengths.mean() or 1)


def _bm25(question: str, mtime: float, n_docs: int, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    postings, lengths, avg = _bm25_index(mtime)
    scores = np.zeros(n_docs, dtype=np.float32)
    for term in set(_tokens(question)):
        plist = postings.get(term)
        if not plist:
            continue
        idf = np.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
        for i, tf in plist:
            scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avg))
    return scores


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search: vector similarity plus BM25 keywords, merged with reciprocal rank fusion."""
    mtime = (INDEX_DIR / "vectors.npy").stat().st_mtime
    vectors, chunks = _load_index(mtime)
    dense = vectors @ _embed([question], is_query=True)[0]
    sparse = _bm25(question, mtime, len(chunks))
    fused: dict[int, float] = {}
    for scores in (dense, sparse):
        for rank, i in enumerate(np.argsort(scores)[::-1][:POOL]):
            if scores[i] > 0:
                fused[int(i)] = fused.get(int(i), 0.0) + 1 / (60 + rank)
    out, seen = [], set()
    for i in sorted(fused, key=fused.get, reverse=True):
        c = chunks[i]
        if c["source"] in EXCLUDE_SOURCES or c["text"] in seen:
            continue
        seen.add(c["text"])
        out.append({**c, "score": float(dense[i])})
        if len(out) == k:
            break
    return out


_dead_until: dict[str, float] = {}  # model -> time before which it is skipped (quota hit)
_answer_cache: dict[str, str] | None = None


def _cache_get(key: str) -> str | None:
    global _answer_cache
    if _answer_cache is None:
        try:
            _answer_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _answer_cache = {}
    return _answer_cache.get(key)


def _cache_put(key: str, text: str) -> None:
    _cache_get(key)  # make sure it is loaded
    _answer_cache[key] = text
    try:
        INDEX_DIR.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_answer_cache), encoding="utf-8")
    except OSError:
        pass


def warmup() -> None:
    """Load the index and the local embedding model so the first question is not slow."""
    if index_exists():
        _load_index((INDEX_DIR / "vectors.npy").stat().st_mtime)
        _embed(["warm up"], is_query=True)


def answer_stream(
    question: str, history: list[dict] | None = None, style: str = "short", use_cache: bool = False
) -> Iterator[str]:
    """Return an iterator of answer text pieces. use_cache: reuse/store answers for self-contained questions."""
    if (chat := chat_local.smalltalk(question)) is not None:
        return iter([chat])
    key = style + "|" + " ".join(question.lower().split())
    if use_cache and (hit := _cache_get(key)):
        def replay() -> Iterator[str]:
            for word in hit.split(" "):
                yield word + " "
                time.sleep(0.012)  # keeps the streaming feel without any API call
        return replay()

    search_query = _to_english(question, history) if _needs_rewrite(question) else question
    sources = retrieve(search_query)
    context = "\n\n".join(s["text"] for s in sources)
    convo = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in (history or [])[-4:])
    length = (
        "Answer in at most 3 short sentences or a few bullets."
        if style == "short"
        else "Give a thorough, well-structured answer with headings or bullets where useful."
    )
    prompt = f"{length}\n\nContext:\n{context}\n\nConversation so far:\n{convo}\nQuestion: {question}"
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2)

    def generate() -> Iterator[str]:
        for model in GEN_MODELS:
            if _dead_until.get(model, 0) > time.time():
                continue
            parts: list[str] = []
            try:
                for chunk in _client().models.generate_content_stream(model=model, contents=prompt, config=config):
                    if chunk.text:
                        parts.append(chunk.text)
                        yield chunk.text
                if use_cache and parts:
                    _cache_put(key, "".join(parts))
                return
            except Exception as e:  # 429 quota / 503 overload / missing key / bad model: try the next model
                log.warning("generation failed on %s: %s", model, e)
                _dead_until[model] = time.time() + (3600 if getattr(e, "code", None) == 429 else 30)
                if parts:
                    return  # keep the partial answer instead of failing the whole reply
        # Every model failed: answer from the retrieved website text so the user still gets help.
        yield chat_local.fallback_answer(question, sources)

    return generate()


def answer(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    return "".join(answer_stream(question, history)), retrieve(question)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks.")
