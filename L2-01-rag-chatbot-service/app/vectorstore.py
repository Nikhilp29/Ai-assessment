"""
Thin wrapper around a local, persistent Chroma collection.

We compute embeddings ourselves (via app.embeddings.embed_texts) and pass
them to Chroma explicitly, rather than using Chroma's built-in embedding
function -- this keeps the embedding backend (local vs Gemini) fully under
our control and swappable in one place.
"""
import uuid
import chromadb

from app.config import get_settings
from app.embeddings import embed_texts

settings = get_settings()

_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection(name=settings.collection_name)


def add_chunks(chunks: list[str], filename: str) -> int:
    """Embed and store a document's chunks. Returns the number of chunks stored."""
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k most relevant chunks for `question`, each as
    {"text": ..., "filename": ..., "distance": ...}."""
    top_k = top_k or settings.top_k
    if _collection.count() == 0:
        return []

    query_embedding = embed_texts([question])[0]
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count()),
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        hits.append({"text": doc, "filename": meta.get("filename", "unknown"), "distance": dist})
    return hits


def document_count() -> int:
    return _collection.count()


def reset_collection() -> None:
    """Delete and recreate the collection -- used by /reset for demos."""
    global _collection
    _client.delete_collection(name=settings.collection_name)
    _collection = _client.get_or_create_collection(name=settings.collection_name)


def list_filenames() -> list[str]:
    """Distinct filenames currently indexed, for display in the UI."""
    if _collection.count() == 0:
        return []
    all_docs = _collection.get()
    filenames = {m.get("filename", "unknown") for m in all_docs.get("metadatas", [])}
    return sorted(filenames)
