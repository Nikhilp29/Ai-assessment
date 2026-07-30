"""
Embedding generation, pluggable between a local model and an API-based one.

- "local": sentence-transformers (all-MiniLM-L6-v2 by default) -- free, runs
  fully offline, but loads a ~90MB model into memory (plus the torch runtime)
  the first time it's used. Fine on a normal machine; can be tight on a
  512MB-RAM free-tier host.
- "gemini": Gemini's text-embedding-004 endpoint -- near-zero local memory,
  but needs network + GEMINI_API_KEY (Gemini's free tier covers this easily
  for a demo's worth of traffic).
"""
from functools import lru_cache
from app.config import get_settings

settings = get_settings()


class EmbeddingConfigError(Exception):
    pass


@lru_cache
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Used both for indexing chunks and embedding
    a user's query (same function, same model, so they live in the same
    vector space)."""
    if settings.embedding_provider == "local":
        model = _local_model()
        return model.encode(texts, convert_to_numpy=True).tolist()
    elif settings.embedding_provider == "gemini":
        return _embed_with_gemini(texts)
    else:
        raise EmbeddingConfigError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. Use 'local' or 'gemini'."
        )


def _embed_with_gemini(texts: list[str]) -> list[list[float]]:
    import google.generativeai as genai

    if not settings.gemini_api_key:
        raise EmbeddingConfigError(
            "EMBEDDING_PROVIDER is 'gemini' but GEMINI_API_KEY is not set."
        )
    genai.configure(api_key=settings.gemini_api_key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        embeddings.append(result["embedding"])
    return embeddings
