<<<<<<< HEAD
# Model Evaluation Dashboard

Run one prompt across multiple LLM providers (Groq, OpenAI, Gemini) at once
and compare latency, estimated cost, and quality side by side. FastAPI
backend + Streamlit dashboard.

Built for OS3 Infotech's AI Engineer Technical Evaluation (Set 3 — L3-12).
Reuses the same LLM-calling pattern as L1-07 and L2-01, but extended to
call several providers per request instead of one configured provider.

## Architecture

```
        POST /benchmark (prompt, providers[], reference_answer?)
                          │
                          ▼
                ┌──────────────────┐
                │   FastAPI layer   │  main.py
                │ (validation,      │
                │  rate limiting)   │
                └────────┬──────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Benchmark runner  │  benchmark.py
                │  loops over each   │
                │  requested provider│
                └────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     ┌─────────┐   ┌─────────┐   ┌─────────┐
     │  Groq    │   │ OpenAI   │   │ Gemini   │   llm_factory.py
     │  call    │   │  call    │   │  call    │   (per-provider,
     │ + timer  │   │ + timer  │   │ + timer  │    explicit provider
     └────┬────┘   └────┬────┘   └────┬────┘    arg, not env-fixed)
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                ┌──────────────────┐
                │  Per-result metrics│  tokenizer.py, pricing.py
                │  latency, est.     │
                │  tokens, est. cost,│
                │  quality score     │
                └──────────────────┘
```

## How it works

1. **Multi-provider LLM factory** (`llm_factory.py`) — unlike L1-07/L2-01
   where one provider is fixed via `LLM_PROVIDER`, this dashboard needs to
   call *several* providers for the same prompt. `call_provider(name, prompt)`
   takes the provider explicitly per call, and `available_providers()`
   reports which ones actually have an API key configured — so the
   dashboard gracefully benchmarks only what's usable and reports the rest
   as unavailable rather than crashing.

2. **Benchmark runner** (`benchmark.py`) — for each requested provider:
   times the call wall-clock, estimates input/output token counts, looks up
   approximate per-token pricing, and (if a reference answer was supplied)
   computes a quality score. Errors from one provider (missing key, rate
   limit, timeout) don't stop the others — each result is independently
   `"status": "ok"` or `"status": "error"`.

3. **Token/cost estimation** (`tokenizer.py`, `pricing.py`) — token counts
   use a simple ~4-characters-per-token heuristic rather than pulling in
   three different real tokenizers (`tiktoken` for OpenAI, Groq's Llama
   tokenizer, Gemini's own) just for an estimate. Costs are computed from a
   small static pricing table. **Both are explicitly labeled "estimated"
   in the API and UI** — the pricing table especially will drift out of
   date and should be checked against each provider's current pricing page
   before being used for real budget decisions (this is called out directly
   in `pricing.py`'s docstring).

4. **Quality scoring** (`benchmark.py::_quality_score`) — a lightweight,
   dependency-free proxy: percentage of the reference answer's distinct
   words that also appear in the model's output. This is lexical overlap,
   not semantic similarity — chosen deliberately to avoid adding an
   embedding model as a dependency purely for scoring on what may be a
   memory-constrained free host. **Good interview talking point**: the
   natural extension here is LLM-as-judge scoring (have one model rate the
   others on a rubric) or embedding-based cosine similarity for a more
   nuanced score — this simple version was chosen to keep the deployment
   light and the scoring logic fully transparent/explainable.

## Project structure

```
model-eval-dashboard/
├── app/
│   ├── main.py         # FastAPI: /benchmark, /health, rate limiting
│   ├── config.py        # environment-based settings (all 3 providers)
│   ├── llm_factory.py    # per-provider call_provider(), available_providers()
│   ├── benchmark.py      # runs the comparison, collects metrics
│   ├── pricing.py        # approximate cost-per-1K-token table (see caveat above)
│   ├── tokenizer.py      # ~4-chars-per-token estimate, provider-agnostic
│   └── schemas.py        # Pydantic request/response models
├── frontend/
│   └── streamlit_app.py  # dashboard: comparison table + latency/cost/quality charts
├── tests/
│   └── test_api.py       # mocked provider calls, no API keys/network needed
├── requirements.txt
├── Dockerfile
├── start.sh              # runs FastAPI (internal :8000) + Streamlit (public port)
├── .env.example
└── README.md
```

## Setup (local)

```bash
cd model-eval-dashboard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in whichever provider keys you want to compare -- you don't need
# all three. The dashboard only benchmarks providers with a key set and
# reports the rest as unavailable.

uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
streamlit run frontend/streamlit_app.py
```
Opens at `http://localhost:8501`.

## Example usage (REST API directly)

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Explain the difference between REST and GraphQL in one sentence.",
        "providers": ["groq", "openai", "gemini"],
        "reference_answer": "REST uses multiple fixed endpoints per resource, while GraphQL exposes a single flexible query endpoint."
      }'
```

Response shape (one entry per provider):
```json
{
  "prompt": "...",
  "results": [
    {
      "provider": "groq",
      "model": "llama-3.1-8b-instant",
      "status": "ok",
      "output": "...",
      "latency_seconds": 0.8,
      "input_tokens_est": 18,
      "output_tokens_est": 24,
      "estimated_cost_usd": 0.000003,
      "quality_score": 72.5
    }
  ]
}
```

If a provider has no key configured, or the call fails (rate limit,
timeout, bad key), that entry comes back with `"status": "error"` and an
`"error"` message instead of stopping the whole benchmark.

## Running tests

```bash
pytest tests/ -v
```

