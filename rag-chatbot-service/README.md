<<<<<<< HEAD
# RAG Chatbot Service

Upload documents, then ask questions answered from their content via
retrieval-augmented generation. FastAPI backend + Streamlit chat UI.

Built for OS3 Infotech's AI Engineer Technical Evaluation (Set 2 — L2-01).
Shares its ingestion/chunking module with L1-07, and its LLM provider
abstraction with both L1-07 and L3-12.

## Architecture

```
   POST /ingest (file)              POST /chat (session_id, message)
          │                                     │
          ▼                                     ▼
  ┌───────────────┐                   ┌──────────────────┐
  │  Ingestion     │                   │  Vector query     │  vectorstore.py
  │  (same as      │                   │  embed question →  │
  │   L1-07)       │                   │  top-k chunks      │
  │  extract text  │                   └─────────┬──────────┘
  │  → chunk       │                             │
  └───────┬────────┘                             ▼
          ▼                            ┌──────────────────┐
  ┌───────────────┐                   │  Prompt builder    │  prompts.py
  │  Embed chunks   │  embeddings.py   │  context + history  │
  │  (local model    │                 │  + question         │
  │   or Gemini API) │                 └─────────┬──────────┘
  └───────┬────────┘                             ▼
          ▼                            ┌──────────────────┐
  ┌───────────────┐                   │   LLM Factory       │  llm_factory.py
  │  Store in Chroma │◄───────────────│  Groq/OpenAI/Gemini  │
  │  (persistent)     │                └─────────┬──────────┘
  └───────────────┘                              ▼
                                        answer + sources, appended
                                        to that session's history
```

## How it works

1. **Ingestion** (`ingestion.py`) — identical module to L1-07: `pypdf`/
   `python-docx` extract text, then a hand-rolled recursive splitter chunks
   it (paragraph → line → sentence → raw-character fallback, with overlap).

2. **Embeddings** (`embeddings.py`) — pluggable between two backends:
   - `local` (default): `sentence-transformers` (`all-MiniLM-L6-v2`), free,
     fully offline, ~90MB model + the `torch` runtime (~200-400MB RAM).
   - `gemini`: Gemini's `text-embedding-004` API, near-zero local memory,
     needs `GEMINI_API_KEY` (free tier is generous enough for a demo).

   **Why two options**: local embeddings avoid any per-request API cost or
   network dependency, which is nice for a demo — but `torch` is heavy, and
   on a memory-constrained free host (e.g. Render's 512MB free tier) it can
   be too much alongside FastAPI + Streamlit in the same container. Setting
   `EMBEDDING_PROVIDER=gemini` before deploying trades a small amount of
   embedding latency for a much lighter container — worth knowing as a
   talking point even if you demo locally with `local`.

3. **Vector store** (`vectorstore.py`) — a local, persistent Chroma
   collection. Embeddings are computed by *us* (via `embeddings.py`) and
   passed to Chroma explicitly, rather than using Chroma's own built-in
   embedder — this keeps the embedding backend (local vs Gemini) fully
   swappable in one place instead of split across two systems.

4. **Retrieval + generation** (`rag.py`) — embeds the user's question with
   the *same* embedding function used at ingest time (same vector space),
   retrieves the top-k (default 4) most relevant chunks, builds a prompt
   that includes those chunks plus recent conversation history, and calls
   the configured LLM. The prompt explicitly instructs the model to answer
   only from the provided context and say so if it doesn't know — reduces
   hallucination on out-of-scope questions.

5. **Session memory** — an in-memory `dict[session_id, history]` (same
   simple pattern used across all three OS3 projects) gives each chat
   session its own conversation continuity. Known limitation, worth stating
   proactively: single-instance only, resets on restart — a Redis-backed
   store would be the production fix.

## Project structure

```
rag-chatbot-service/
├── app/
│   ├── main.py         # FastAPI: /ingest, /chat, /health, /status, /reset
│   ├── config.py        # environment-based settings
│   ├── ingestion.py      # PDF/DOCX extraction + chunking (shared w/ L1-07)
│   ├── embeddings.py     # pluggable local/Gemini embedding backend
│   ├── vectorstore.py    # Chroma wrapper: add_chunks, query, reset
│   ├── prompts.py        # RAG prompt template (context + history + question)
│   ├── rag.py            # retrieval + generation orchestration + session memory
│   ├── llm_factory.py    # Groq/OpenAI/Gemini provider abstraction (shared)
│   └── schemas.py        # Pydantic request/response models
├── frontend/
│   └── streamlit_app.py  # chat UI: sidebar upload, chat_message history, sources
├── tests/
│   └── test_api.py       # full pipeline tested with mocked embeddings + LLM
├── sample_docs/
├── requirements.txt
├── Dockerfile
├── start.sh              # runs FastAPI (internal :8000) + Streamlit (public port)
├── .env.example
└── README.md
```

## Setup (local)

```bash
cd rag-chatbot-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER + matching API key
# EMBEDDING_PROVIDER=local works out of the box (downloads the model on
# first use); switch to "gemini" if you'd rather not pull in torch

uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
streamlit run frontend/streamlit_app.py
```
Opens at `http://localhost:8501`. Upload a PDF/DOCX in the sidebar, then
chat about it.

## Example usage (REST API directly)

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_docs/sample_report.docx"

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What happened to cloud costs this quarter?"}'
```

## Running tests

```bash
pytest tests/ -v
```
