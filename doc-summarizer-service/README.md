# Document Summarization Service

A REST API that accepts a PDF or DOCX file and returns a summary in one of
three configurable styles: `brief`, `detailed`, or `bullet-points`. Handles
arbitrarily long documents via a map-reduce summarization pipeline.

Built for OS3 Infotech's AI Engineer Technical Evaluation (Set 1 — L1-07).

## Architecture

```
                POST /summarize (file + style)
                          │
                          ▼
                ┌──────────────────┐
                │   FastAPI layer   │  main.py
                │  (validation,     │
                │   rate limiting,  │
                │   error mapping)  │
                └────────┬──────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Ingestion      │  ingestion.py
                │  PDF/DOCX → text  │
                │  → chunk splitter │
                └────────┬──────────┘
                         │
             1 chunk ────┼──── 2+ chunks
                │                  │
                ▼                  ▼
        ┌───────────────┐  ┌──────────────────────┐
        │ Direct summary │  │   Map-Reduce          │  summarizer.py
        │  (single call) │  │ MAP: summarize each   │
        └───────┬────────┘  │      chunk            │
                │           │ REDUCE: combine into   │
                │           │   final styled summary │
                │           └──────────┬─────────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────┐
                │   LLM Factory     │  llm_factory.py
                │ Groq/OpenAI/Gemini│
                │ (env-selected)    │
                └───────────────────┘
```

## How it works

1. **Ingestion** (`ingestion.py`) — `pypdf` extracts text from PDFs,
   `python-docx` extracts paragraph text from Word files. The result is one
   plain-text string per document.

2. **Chunking** — a hand-implemented version of the same idea behind
   LangChain's `RecursiveCharacterTextSplitter`: try splitting on paragraph
   breaks first, fall back to line breaks, then sentence boundaries, then
   raw character cuts — only escalating to a harder split when a piece is
   still too large. Consecutive chunks share a small overlap (default 400
   chars) so context isn't lost right at a chunk boundary. It's implemented
   by hand rather than imported so the exact behavior is visible and easy to
   explain/modify; swapping in LangChain's splitter is a one-line change in
   `ingestion.py` if preferred.

3. **Map-Reduce summarization** (`summarizer.py`):
   - If the whole document fits in one chunk, it's summarized directly in
     the requested style — no need to pay for two LLM calls.
   - Otherwise: **Map** — each chunk is summarized independently (style-
     neutral, fact-preserving prompt). **Reduce** — all chunk summaries are
     concatenated and summarized *again*, this time with the style-specific
     instruction (`brief` / `detailed` / `bullet-points`). Applying style
     only at the reduce step means the same map-step summaries could be
     reused to produce a different style without re-processing the whole
     document.

4. **LLM Factory** (`llm_factory.py`) — a single `call_llm(prompt)` function
   fronts three providers (Groq, OpenAI, Gemini), selected via the
   `LLM_PROVIDER` environment variable. Everything else in the codebase is
   provider-agnostic, which also makes this service reusable as the backend
   for the Model Evaluation Dashboard (Set 3) later.

5. **REST layer** (`main.py`) — validates file type and size before doing
   any expensive work, applies a simple in-memory per-IP rate limit, and
   maps internal exceptions to correct HTTP status codes (400 for bad file
   type, 422 for an unreadable/empty document, 502 if the upstream LLM call
   fails, 429 if rate-limited).

## Why this design

- **Map-reduce instead of a giant single prompt**: LLMs have finite context
  windows and quality degrades on very long inputs ("lost in the middle").
  Map-reduce lets the service handle a 200-page PDF the same way it handles
  a one-paragraph note, at the cost of extra LLM calls.
- **Style applied at reduce time, not map time**: keeps the map step
  reusable and cheaper to change later (e.g., add a 4th style without
  touching how chunks are summarized).
- **Provider abstraction**: enterprise LLM services shouldn't be locked to
  one vendor — this mirrors what Set 3's L3-08 (multi-provider config)
  asks for, and lets you swap providers per cost/latency/privacy needs.
- **Hand-rolled splitter**: demonstrates the underlying algorithm rather
  than hiding it behind a library import — useful to be able to explain in
  an interview, and removes a large dependency (full `langchain`) for a
  service that only needs the splitting logic.

## Project structure

```
doc-summarizer-service/
├── app/
│   ├── main.py          # FastAPI app: routes, validation, rate limiting
│   ├── config.py        # environment-based settings
│   ├── schemas.py        # Pydantic request/response models
│   ├── ingestion.py      # PDF/DOCX extraction + chunking
│   ├── prompts.py        # style-specific prompt templates
│   ├── summarizer.py     # map-reduce orchestration
│   └── llm_factory.py    # Groq/OpenAI/Gemini provider abstraction
├── frontend/
│   ├── streamlit_app.py  # Streamlit UI (upload, style picker, results)
│   └── requirements.txt  # frontend-only deps (streamlit, requests)
├── tests/
│   └── test_api.py       # API tests with a mocked LLM (no key needed)
├── sample_docs/
│   └── sample_report.docx
├── requirements.txt
├── .env.example
└── README.md
```

## Frontend (Streamlit)

A thin UI on top of the API — it does no parsing or LLM work itself, it just
calls `POST /summarize` and renders the JSON response. Kept as a **separate**
app/process from the backend (talks over HTTP, not imported directly) so the
API remains independently testable/deployable, which is closer to how a
real production setup would be split.

Features:
- File uploader (PDF/DOCX) + style selector (`brief` / `detailed` / `bullet-points`)
- Sidebar backend health check — shows connection status, active provider, and model
- Results view: rendered summary, chunk count, map-reduce flag, processing
  time, model — plus a raw-JSON expander and a "download as .txt" button
- Surfaces the backend's actual error detail (400/422/429/502) instead of a
  generic failure message

**Run it** (with the backend already running on port 8000):

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`. If your backend runs somewhere other than
`localhost:8000`, set `SUMMARIZER_API_URL` before launching:

```bash
SUMMARIZER_API_URL=http://127.0.0.1:8000 streamlit run frontend/streamlit_app.py
```

## Setup

```bash
cd doc-summarizer-service
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key
# Groq has a generous free tier and is the default — get a key at console.groq.com

uvicorn app.main:app --reload --port 8000
```

API docs (Swagger UI) will be at `http://localhost:8000/docs`.

## Example usage

```bash
curl -X POST http://localhost:8000/summarize \
  -F "file=@sample_docs/sample_report.docx" \
  -F "style=bullet-points"
```

Sample response (against the included `sample_report.docx`, a mock
quarterly infra report):

```json
{
  "filename": "sample_report.docx",
  "style": "bullet-points",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "chunk_count": 1,
  "used_map_reduce": false,
  "processing_time_seconds": 0.01,
  "summary": "- 99.95% uptime, beating the 99.9% SLA target\n- Cloud spend cut ~18% via right-sizing and cold storage tiering\n- MTTD dropped 14min -> 6min; MTTR dropped 52min -> 31min after new observability stack\n- Q3 plans: multi-region failover test, 2 new hires, service mesh evaluation"
}
```

The same document with `style=brief`:

```json
{
  "summary": "The infrastructure team exceeded its uptime SLA in Q2 2026, cut cloud costs by 18%, and significantly improved incident response times after deploying new observability tooling."
}
```

And with `style=detailed`:

```json
{
  "summary": "The cloud infrastructure team delivered a strong Q2 2026, hitting 99.95% uptime against a 99.9% SLA target with zero customer-facing incidents across three maintenance windows. Cost optimization work — instance right-sizing and cold-tier storage migration — cut monthly cloud spend by about 18%, while a managed database migration reduced on-call overhead. A new observability stack roughly halved both detection and resolution times for incidents. For Q3, the team is planning a multi-region failover exercise, onboarding two engineers, and evaluating a service mesh, with budget requested for additional load-testing infrastructure."
}
```

For a document long enough to trigger map-reduce (e.g. a 50-page PDF), the
response looks the same but with `"chunk_count": 6` (or however many) and
`"used_map_reduce": true`.

## Running tests

```bash
pytest tests/ -v
```

Tests mock the LLM call so they run instantly with no API key and no
network access — useful for CI, and for verifying the chunking/map-reduce
branching logic in isolation from any provider's actual behavior.

## Possible extensions (good talking points for the interview)

- Swap the in-memory rate limiter for Redis to work across multiple
  service instances.
- Add streaming responses (SSE) so the client sees partial summaries as
  each map-step chunk finishes, instead of waiting for the whole pipeline.
- Add a `/summarize/async` endpoint backed by a task queue (Celery/RQ) for
  very large documents, returning a job ID to poll instead of blocking the
  HTTP request.
- Cache chunk-level summaries by content hash so re-summarizing the same
  document in a different style skips the map step entirely.
