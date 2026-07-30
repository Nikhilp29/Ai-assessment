from pydantic import BaseModel


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_store: int


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    filename: str
    preview: str
    distance: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    documents_indexed: int


class StatusResponse(BaseModel):
    indexed_files: list[str]
    total_chunks: int
