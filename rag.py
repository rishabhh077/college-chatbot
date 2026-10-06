"""RAG pipeline: load docs -> chunk -> embed (local) -> retrieve -> answer."""
import json
import os
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

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
CACHE_FILE = INDEX_DIR / "answer_cache.json"
# Comma-separated; if one is rate-limited (free-tier daily caps) the next is tried.
GEN_MODELS = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-3.7-flash,gemini-3.8-flash").split(",")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = """You are the official AI assistant of LIET (Lloyd Institute of Engineering & Technology), Greater Noida.

How to answer:
1. Use the provided LIET website/document excerpts as your primary source. Answer as LIET's own assistant: say "LIET" or "our college" and never describe it as "a private college", "private engineering colleges" or "colleges in Greater Noida" in general.
2. The excerpts include blog articles that use generic phrases like "top private colleges" or mention other institutions. Treat those as being about LIET, and never present other colleges' details as LIET's.
3. If the excerpts do not contain the answer and the question is about LIET specifics (fees, dates, names, phone numbers, policies), do not guess. Say you don't have that detail and give LIET's contact: admissions@liet.in, +91 9821582662.
4. If the question is general (study advice, career, technology, entrance exams, or anything not specific to LIET), answer helpfully from your own knowledge, starting with "Not from LIET's website:". Where it fits, connect the answer back to LIET's relevant programs.
5. Be concise and friendly. Do not mention file names, sources, page numbers or the word "context/excerpts" in your answer; the app shows sources separately."""


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


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    vectors, chunks = _load_index((INDEX_DIR / "vectors.npy").stat().st_mtime)
    q = _embed([question], is_query=True)[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


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

    sources = retrieve(question)
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
        last_err = None
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
            except (genai.errors.ServerError, genai.errors.ClientError) as e:
                last_err = e  # 429 quota / 503 overload: skip this model for a while, try the next
                _dead_until[model] = time.time() + (3600 if e.code == 429 else 30)
                if parts:
                    raise  # already streamed part of an answer; do not restart on another model
        raise last_err or RuntimeError("All models are rate-limited. Try again later.")

    return generate()


def answer(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    return "".join(answer_stream(question, history)), retrieve(question)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks.")
