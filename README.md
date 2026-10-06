# College Query Chatbot (RAG)

Answers student questions from your college's own documents using Retrieval-Augmented Generation.

**Pipeline:** docs (PDF/TXT) → chunks → Gemini embeddings → cosine search (NumPy) → Gemini answer grounded in the top chunks, with sources shown.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your free key from https://aistudio.google.com/apikey
```

## Run
1. Put your PDFs / .txt files in `docs/` (a sample FAQ is included).
2. `streamlit run app.py`, then click **Rebuild index** in the sidebar (or run `python rag.py`).
3. Ask questions.

## Files
- `rag.py` – loading, chunking, embedding, retrieval, answering
- `app.py` – Streamlit chat UI
- `docs/` – knowledge base; `index/` – generated embeddings (git-ignored)
