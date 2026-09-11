import sqlite3
import requests

# ISSUE 1 (Security — should be CRITICAL): hardcoded API key committed to source
GROQ_API_KEY = "gsk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"

DB_PATH = "chat_sessions.db"


def get_user_history(user_id):
    """Fetch a user's past chat messages for context injection into the RAG prompt."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ISSUE 2 (Security — should be CRITICAL): raw string formatting into SQL,
    # classic SQL injection if user_id ever comes from an unvalidated request param
    query = "SELECT message, response FROM history WHERE user_id = '" + user_id + "'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows


def embed_messages(messages):
    """Get embeddings for a batch of chat messages before storing in the vector store."""
    embeddings = []
    # ISSUE 3 (Performance — should be MEDIUM/HIGH): N+1 API calls in a loop
    # instead of one batched embedding request -- will be painfully slow and
    # burn through free-tier rate limits fast on any real conversation history.
    for msg in messages:
        resp = requests.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"input": msg, "model": "text-embedding-3-small"},
        )
        embeddings.append(resp.json()["data"][0]["embedding"])
    return embeddings


def get_last_n_messages(user_id, n):
    """Return the last n messages for a user, most recent first."""
    history = get_user_history(user_id)
    # ISSUE 4 (Bug — should be MEDIUM): no check for empty history or n > len(history);
    # also silently returns wrong order on some inputs instead of raising/handling it
    return history[-n:]