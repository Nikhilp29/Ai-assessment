import os
import sqlite3
from contextlib import contextmanager

import requests

# FIX 1: no hardcoded key — loaded from environment instead
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

DB_PATH = "chat_sessions.db"


@contextmanager
def _get_connection():
    # FIX 5: connection is still opened per call for simplicity/thread-safety,
    # but centralized here as a single reusable helper instead of duplicated
    # open/close logic in every function.
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_user_history(user_id: str):
    """Fetch a user's past chat messages for context injection into the RAG prompt."""
    if not user_id:
        raise ValueError("user_id is required")

    with _get_connection() as conn:
        cursor = conn.cursor()
        # FIX 2: parameterized query — no string concatenation, immune to SQL injection
        cursor.execute(
            "SELECT message, response FROM history WHERE user_id = ?",
            (user_id,),
        )
        return cursor.fetchall()


def embed_messages(messages: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of chat messages before storing in the vector store."""
    if not messages:
        return []

    # FIX 3: single batched request instead of one HTTP call per message
    resp = requests.post(
        "https://api.groq.com/openai/v1/embeddings",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"input": messages, "model": "text-embedding-3-small"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [item["embedding"] for item in data]


def get_last_n_messages(user_id: str, n: int):
    """Return the last n messages for a user, most recent first."""
    if n <= 0:
        return []

    history = get_user_history(user_id)
    # FIX 4: explicit handling for empty history and n > len(history),
    # and reversed so most-recent is actually first as the docstring promises
    if not history:
        return []
    return list(reversed(history[-n:]))