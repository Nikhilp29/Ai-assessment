"""
Environment-based configuration.

All secrets/config are loaded from environment variables (via a .env file in
local dev) so nothing sensitive is hard-coded in source. This is the same
pattern the L3-08 "Secure Multi-Provider" task expects, so it's written to be
easy to extend later.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Which LLM backend to use: "groq" | "openai" | "gemini"
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # Provider credentials (only the one you use needs to be set)
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # Model names per provider (override in .env if you want a different model)
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Chunking parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))       # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters

    # Upload limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    # Basic request throttling (very simple in-memory limiter)
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
