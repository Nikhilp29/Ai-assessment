"""
FastAPI service exposing document summarization.

Endpoints:
  POST /summarize  - upload a PDF/DOCX, get back a styled summary
  GET  /health      - liveness + current provider/model info
"""
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.ingestion import extract_text, UnsupportedFileTypeError, EmptyDocumentError
from app.summarizer import summarize_document
from app.llm_factory import current_model_name, LLMConfigError, LLMCallError
from app.schemas import SummaryResponse, HealthResponse, SummaryStyle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("doc-summarizer")

settings = get_settings()

app = FastAPI(
    title="Document Summarization Service",
    description="Upload a PDF/DOCX and get a configurable-style summary (brief / detailed / bullet-points).",
    version="1.0.0",
)

# --- very small in-memory rate limiter (per client IP) ---------------------
# Good enough for a single-instance demo; swap for Redis-backed limiting
# (e.g. slowapi + redis) before running multiple instances in production.
_request_log: dict[str, deque] = defaultdict(deque)


def _rate_limit_ok(client_ip: str) -> bool:
    now = time.time()
    window = _request_log[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return False
    window.append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again in a minute."},
        )
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        model=current_model_name(),
    )


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(
    file: UploadFile = File(...),
    style: SummaryStyle = Form(SummaryStyle.brief),
):
    # --- basic request validation ---
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed is {settings.max_file_size_mb}MB.",
        )

    start = time.time()
    logger.info(f"Received '{file.filename}' ({size_mb:.2f}MB), style={style.value}")

    # --- extraction ---
    try:
        text = extract_text(file.filename, file_bytes)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # --- summarization ---
    try:
        summary, chunk_count, used_map_reduce = summarize_document(text, style.value)
    except LLMConfigError as exc:
        logger.error(f"LLM config error: {exc}")
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: {exc}")
    except LLMCallError as exc:
        logger.error(f"LLM call failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")

    elapsed = time.time() - start
    logger.info(
        f"Summarized '{file.filename}' in {elapsed:.2f}s "
        f"({chunk_count} chunk(s), map_reduce={used_map_reduce})"
    )

    return SummaryResponse(
        filename=file.filename,
        style=style,
        provider=settings.llm_provider,
        model=current_model_name(),
        chunk_count=chunk_count,
        used_map_reduce=used_map_reduce,
        processing_time_seconds=round(elapsed, 2),
        summary=summary,
    )
