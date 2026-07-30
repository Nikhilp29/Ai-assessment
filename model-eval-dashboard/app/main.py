"""
Model Evaluation Dashboard service.

Endpoints:
  POST /benchmark  - run one prompt across multiple LLM providers, compare
                      latency, estimated cost, and (optional) quality score
  GET  /health       - liveness + which providers have keys configured
"""
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import RATE_LIMIT_PER_MINUTE
from app.llm_factory import available_providers
from app.benchmark import run_benchmark
from app.schemas import BenchmarkRequest, BenchmarkResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model-eval-dashboard")

app = FastAPI(
    title="Model Evaluation Dashboard",
    description="Benchmark multiple LLM providers on the same prompt: latency, estimated cost, and quality.",
    version="1.0.0",
)

VALID_PROVIDERS = {"groq", "openai", "gemini"}

# --- simple in-memory per-IP rate limiter (same pattern as L1-07/L2-01) ---
_request_log: dict[str, deque] = defaultdict(deque)


def _rate_limit_ok(client_ip: str) -> bool:
    now = time.time()
    window = _request_log[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        return False
    window.append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."})
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", available_providers=available_providers())


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    providers = request.providers or available_providers()

    unknown = [p for p in providers if p not in VALID_PROVIDERS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown provider(s): {unknown}. Use groq/openai/gemini.")

    if not providers:
        raise HTTPException(
            status_code=500,
            detail="No LLM providers available -- no API keys are configured on the server.",
        )

    logger.info(f"Running benchmark across {providers} for prompt of length {len(request.prompt)}")
    results = run_benchmark(request.prompt, providers, request.reference_answer)

    return BenchmarkResponse(prompt=request.prompt, results=results)
