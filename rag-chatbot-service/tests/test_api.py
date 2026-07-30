"""
Tests mock both embed_texts() and call_llm() so they run with no network
access and no API keys -- useful for CI, and for verifying the ingest ->
chunk -> embed -> store -> retrieve -> prompt -> generate pipeline wiring
independent of any real embedding model or LLM provider.

The mock embedding is a simple deterministic bag-of-words vector so that
semantically similar mock documents actually retrieve as "close" in the
test below -- good enough to prove the retrieval wiring works without a
real model.
"""
import io
import uuid
import docx
import pytest
from fastapi.testclient import TestClient

from app import main


def _fake_embed(texts):
    """Deterministic fake embedding: a small vector based on word overlap
    with a fixed vocabulary, so related mock texts end up near each other."""
    vocab = ["cloud", "cost", "uptime", "cat", "mat", "weather", "rain"]
    vectors = []
    for text in texts:
        lower = text.lower()
        vectors.append([float(lower.count(word)) for word in vocab])
    return vectors


@pytest.fixture(autouse=True)
def mock_backends(monkeypatch):
    monkeypatch.setattr("app.vectorstore.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "app.rag.call_llm",
        lambda prompt, **kwargs: f"[FAKE ANSWER based on {len(prompt)} char prompt]",
    )
    # each test gets an isolated collection so tests don't leak into each other
    from app.vectorstore import reset_collection
    reset_collection()


@pytest.fixture
def client():
    return TestClient(main.app)


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_then_chat(client):
    content = _make_docx_bytes(["Cloud costs were reduced by 18 percent this quarter."])
    files = {"file": ("report.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_added"] >= 1
    assert body["total_chunks_in_store"] >= 1

    session_id = str(uuid.uuid4())
    r2 = client.post("/chat", json={"session_id": session_id, "message": "What happened to cloud costs?"})
    assert r2.status_code == 200
    chat_body = r2.json()
    assert "FAKE ANSWER" in chat_body["answer"]
    assert chat_body["session_id"] == session_id
    assert len(chat_body["sources"]) >= 1


def test_chat_without_ingest_returns_422(client):
    r = client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": "Anything in the docs?"})
    assert r.status_code == 422


def test_chat_empty_message_returns_400(client):
    content = _make_docx_bytes(["Some content."])
    files = {"file": ("x.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    client.post("/ingest", files=files)
    r = client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": "   "})
    assert r.status_code == 400


def test_unsupported_file_type_returns_400(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 400


def test_status_lists_indexed_files(client):
    content = _make_docx_bytes(["Some content about uptime."])
    files = {"file": ("uptime.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    client.post("/ingest", files=files)
    r = client.get("/status")
    assert r.status_code == 200
    assert "uptime.docx" in r.json()["indexed_files"]


def test_reset_clears_store(client):
    content = _make_docx_bytes(["Content to be reset."])
    files = {"file": ("temp.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    client.post("/ingest", files=files)
    assert client.get("/health").json()["documents_indexed"] >= 1

    r = client.post("/reset")
    assert r.status_code == 200
    assert client.get("/health").json()["documents_indexed"] == 0
