import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM (generation) provider -- same pattern as L1-07
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Embedding provider is SEPARATE from the generation provider:
    #   "local"  -> sentence-transformers, all-MiniLM-L6-v2, runs on CPU, no API cost,
    #               but adds ~200-400MB RAM -- fine locally, tight on Render's free
    #               512MB tier.
    #   "gemini" -> Gemini's text-embedding-004 API, near-zero RAM, needs GEMINI_API_KEY
    #               (free tier). Recommended when deploying to a small free instance.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking (reused pattern from L1-07)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Chroma persistence directory
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_data")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

    # Chat memory: how many past turns to include in the prompt
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "4"))

    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
