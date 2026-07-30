"""
RAG orchestration: retrieve relevant chunks, build a context-stuffed prompt,
call the LLM, and track per-session conversation history in memory.
"""
from app.config import get_settings
from app.vectorstore import query as vector_query
from app.llm_factory import call_llm
from app.prompts import build_rag_prompt

settings = get_settings()

# session_id -> list of (role, content) tuples. In-memory only -- resets on
# restart. Fine for a demo; swap for Redis/a DB to persist across restarts
# or scale beyond a single instance.
_sessions: dict[str, list[tuple[str, str]]] = {}


def get_history(session_id: str) -> list[tuple[str, str]]:
    return _sessions.get(session_id, [])


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def chat(session_id: str, message: str) -> tuple[str, list[dict]]:
    """Returns (answer, sources)."""
    history = _sessions.setdefault(session_id, [])

    hits = vector_query(message, top_k=settings.top_k)
    context_chunks = [h["text"] for h in hits]

    recent_history = history[-(settings.max_history_turns * 2):]
    prompt = build_rag_prompt(message, context_chunks, recent_history)

    answer = call_llm(prompt)

    history.append(("user", message))
    history.append(("assistant", answer))

    sources = [
        {"filename": h["filename"], "preview": h["text"][:200], "distance": round(h["distance"], 4)}
        for h in hits
    ]
    return answer, sources
