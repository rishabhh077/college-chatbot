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
4. If the question is general (study advice, career, technology, entrance exams, or anything not specific to LIET), answer helpfully from your own knowledge (including help with writing emails, explaining concepts, coding or study doubts). Say that it is general advice, not from LIET's website, only where it could be mistaken for official LIET information. Where it fits, connect the answer back to LIET's relevant programs.
5. Never start a reply with "Sorry" and never apologise. For greetings or small talk (hi, thanks, how are you), reply warmly in the user's language and offer help with LIET topics. If a question is vague, ask one short clarifying question.
6. Be concise and friendly. Do not mention file names, sources, page numbers or the word "context/excerpts" in your answer; the app shows sources separately.

Hard questions:
9. For multi-part questions, answer each part the excerpts cover and say plainly which part the website does not state. Never fill a gap with guessed numbers, percentages, company names, rules or dates.
10. If the excerpts give different figures for the same thing (fees, placements, eligibility), show each figure with the session/year it belongs to and tell the student to confirm the current one with admissions. Do not claim one page is more official unless a page says so.
11. For cost or comparison questions, use only published figures, show the arithmetic, list the charges the website does not publish, and mark the total as an estimate to be confirmed.
12. Rules set by the affiliating university (AKTU), AICTE or the government (exam rules, degree award, attendance norms, scholarships) are not LIET's own. Say so and point to the official document or the admissions contact.
13. Questions about what the website "hides", contradicts or advertises vs. regulations cannot be answered from it. Say that, then list what a student should verify and with whom.

How to talk (very important):
6a. Chat like a friendly person on WhatsApp, not a formal robot: simple everyday words, short replies for simple questions, more detail only when the user needs it. Match the user's tone and formality. Do not call everyone "bhai" or "bro", use slang only if the user does, use emojis sparingly, do not repeat the user's question back, and ask a follow-up question only when you genuinely cannot proceed.
6b. Understand meaning, not spelling. Users type fast and informally in English, Hindi (Devanagari or Roman letters) and Hinglish (mixed), with shortcuts, typos, stretched letters, emojis and half sentences (e.g. "muje ye smjh nhi aya", "mera login nhi hora", "fees kitni h yrr", "pls short mein btao"). Work out what they mean from the whole sentence and the conversation, never ask them to retype or fix their spelling, and never correct their grammar unless they ask.
6c. Reply in the language and script the user is using, and keep it consistent with their last messages:
   - English -> natural, simple English.
   - Hindi in Devanagari -> Hindi in Devanagari, everyday words.
   - Hindi typed in Roman letters -> Roman Hindi (Hinglish), the way a friendly Indian student would type. Never mistake it for English and never switch it to Devanagari.
   - A mix of Hindi and English -> a natural Hinglish mix, not formal Hindi and not formal English.
   If the user asks for a language, script or tone ("hindi mein batao but english letters mein", "in simple English", "thoda professional", "short krdo", "easy language mein"), follow it and keep it until they change it.
6d. Use the conversation for follow-ups. "ye kya hota h", "isko short kro", "make it professional", "aur batao", "and that?" refer to your previous answer or the user's earlier message: act on that. If there is nothing earlier to refer to, ask what they mean in one short line.
7. The excerpts are in English. Translate facts into the reply language, but keep names, course names (B.Tech, MBA), numbers, amounts (Rs / INR), emails and phone numbers exactly as they are.
8. Everything else you say (for example "I don't have that detail") must also be in the reply language."""


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


def _search_query(question: str, history: list[dict] | None) -> str | None:
    """Turn a casual message (any spelling, Hindi/Hinglish/English, with follow-up context) into a standalone
    English search query. Returns None when no website lookup is needed (small talk, 'make it shorter', ...)."""
    convo = _convo(history, 4, 500)
    prompt = (
        "You prepare a search query for the website of LIET (Lloyd Institute of Engineering & Technology, "
        "Greater Noida). The user's latest message may be English, Hindi (Devanagari), Roman Hindi or Hinglish, "
        "with slang, typos, abbreviations and emojis. Understand what they mean and resolve words like 'ye', "
        "'isko', 'it', 'that' from the conversation. Output one short standalone English search query. "
        "If the message needs no website lookup (greeting, thanks, or only asks to shorten, rephrase, translate, "
        "change tone or language of the previous answer, or is about something unrelated to LIET), output exactly NONE. "
        "Output only the query or NONE.\n\n"
        f"Conversation:\n{convo}\nLatest message: {question}"
    )
    config = types.GenerateContentConfig(temperature=0)
    for model in GEN_MODELS:
        if _dead_until.get(model, 0) > time.time():
            continue
        try:
            text = (_client().models.generate_content(model=model, contents=prompt, config=config).text or "").strip()
            if not text:
                return question
            return None if text.upper().startswith("NONE") else text
        except Exception as e:  # quota, overload, bad key/model name: fall back to keyword expansion
            _dead_until[model] = time.time() + (3600 if getattr(e, "code", None) == 429 else 30)
            log.warning("query rewrite failed on %s: %s", model, e)
    return chat_local.expand_query(question)


def _convo(history: list[dict] | None, n: int, limit: int) -> str:
    """Last n messages as 'User: ... / Assistant: ...' lines, each clipped to limit characters."""
    return "".join(f"{m['role'].capitalize()}: {m['content'][:limit]}\n" for m in (history or [])[-n:])


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
    key = style + "|" + " ".join(question.lower().split())
    if use_cache and (hit := _cache_get(key)):
        def replay() -> Iterator[str]:
            for word in hit.split(" "):
                yield word + " "
                time.sleep(0.012)  # keeps the streaming feel without any API call
        return replay()

    search_query = _search_query(question, history)
    sources = retrieve(search_query) if search_query else []
    context = "\n\n".join(s["text"] for s in sources) or "(no website lookup needed for this message)"
    convo = _convo(history, 6, 1500)
    length = (
        "Keep it short: at most 3 short sentences or a few bullets, unless the user asks for more detail."
        if style == "short"
        else "Give a thorough, well-structured answer with headings or bullets where useful."
    )
    prompt = f"{length}\n\nContext:\n{context}\n\nConversation so far:\n{convo}\nUser's latest message (reply to this, in their language and style): {question}"
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.4)

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
        # Every model failed: small talk gets a canned friendly reply, anything else an answer built from the website text.
        yield chat_local.smalltalk(question) or chat_local.fallback_answer(question, sources or retrieve(chat_local.expand_query(question)))

    return generate()


def answer(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    return "".join(answer_stream(question, history)), retrieve(question)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks.")
