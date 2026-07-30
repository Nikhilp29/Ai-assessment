"""
Tests use a mocked LLM call so they run without any real API key/network —
useful for CI and for quickly verifying the pipeline logic (chunking,
map-reduce branching, error handling) independent of any provider.
"""
import io
import docx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import SummaryStyle


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    def fake_call_llm(prompt, max_tokens=1024, temperature=0.3):
        return f"[FAKE SUMMARY for prompt of length {len(prompt)}]"

    monkeypatch.setattr("app.summarizer.call_llm", fake_call_llm)
    monkeypatch.setattr("app.main.current_model_name", lambda: "fake-model")


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
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_summarize_small_docx_no_map_reduce(client):
    content = _make_docx_bytes(["This is a short test document about cats."])
    files = {"file": ("cats.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/summarize", files=files, data={"style": SummaryStyle.brief.value})
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 1
    assert body["used_map_reduce"] is False
    assert "FAKE SUMMARY" in body["summary"]


def test_summarize_large_docx_triggers_map_reduce(client):
    # ~10 paragraphs of filler, each long enough to force multiple chunks
    long_paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
    content = _make_docx_bytes([long_paragraph] * 5)
    files = {"file": ("big.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/summarize", files=files, data={"style": SummaryStyle.detailed.value})
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] > 1
    assert body["used_map_reduce"] is True


def test_unsupported_file_type_returns_400(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    response = client.post("/summarize", files=files, data={"style": "brief"})
    assert response.status_code == 400


def test_empty_docx_returns_422(client):
    content = _make_docx_bytes([])
    files = {"file": ("empty.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/summarize", files=files, data={"style": "brief"})
    assert response.status_code == 422
