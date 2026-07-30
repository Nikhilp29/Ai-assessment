"""
RAG Chatbot service.

Endpoints:
  POST /ingest  - upload a PDF/DOCX, chunk it, embed it, store in Chroma
  POST /chat     - ask a question, get an answer grounded in ingested docs
  GET  /health    - liveness + config info
  GET  /status    - which files are indexed, how many chunks total
  POST /reset     - clear the vector store (useful for demos)
"""
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.config import get_settings
from app.ingestion import extract_text, recursive_character_split, UnsupportedFileTypeError, EmptyDocumentError
from app.vectorstore import add_chunks, document_count, reset_collection, list_filenames
from app.rag import chat as rag_chat, clear_session
from app.llm_factory import LLMConfigError, LLMCallError
from app.embeddings import EmbeddingConfigError
from app.schemas import (
    IngestResponse, ChatRequest, ChatResponse, HealthResponse, StatusResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-chatbot")

settings = get_settings()

app = FastAPI(
    title="RAG Chatbot Service",
    description="Upload documents, then ask questions answered from their content via retrieval-augmented generation.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        documents_indexed=document_count(),
    )


@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(indexed_files=list_filenames(), total_chunks=document_count())


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed is {settings.max_file_size_mb}MB.",
        )

    try:
        text = extract_text(file.filename, file_bytes)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    chunks = recursive_character_split(text, settings.chunk_size, settings.chunk_overlap)

    try:
        added = add_chunks(chunks, file.filename)
    except EmbeddingConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Embedding configuration error: {exc}")

    logger.info(f"Ingested '{file.filename}': {added} chunks added.")

    return IngestResponse(
        filename=file.filename,
        chunks_added=added,
        total_chunks_in_store=document_count(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if document_count() == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents have been ingested yet. Upload a document via /ingest first.",
        )

    try:
        answer, sources = rag_chat(request.session_id, request.message)
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: {exc}")
    except LLMCallError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")
    except EmbeddingConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Embedding configuration error: {exc}")

    return ChatResponse(session_id=request.session_id, answer=answer, sources=sources)


@app.post("/reset")
def reset(session_id: str | None = None):
    reset_collection()
    if session_id:
        clear_session(session_id)
    return {"status": "reset complete"}
