"""
Tests mock call_provider() so they run instantly with no API keys and no
network -- verifies the benchmark orchestration (latency measurement,
token/cost estimation, quality scoring, error handling per-provider)
independent of any real LLM's actual behavior.
"""
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    return TestClient(main.app)


def _fake_call_provider(provider, prompt, max_tokens=512, temperature=0.3):
    responses = {
        "groq": "REST uses fixed endpoints and resources. GraphQL uses a single flexible query endpoint.",
        "openai": "REST exposes multiple resource endpoints; GraphQL exposes one endpoint with flexible queries.",
        "gemini": "The main difference is REST has multiple endpoints while GraphQL has one flexible endpoint.",
    }
    return responses.get(provider, "generic response")


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setattr("app.benchmark.call_provider", _fake_call_provider)
    monkeypatch.setattr("app.main.available_providers", lambda: ["groq", "openai", "gemini"])
    monkeypatch.setattr("app.benchmark.MODEL_NAMES", {"groq": "llama-3.1-8b-instant", "openai": "gpt-4o-mini", "gemini": "gemini-1.5-flash"})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["available_providers"]) == {"groq", "openai", "gemini"}


def test_benchmark_all_providers(client):
    r = client.post("/benchmark", json={
        "prompt": "Explain REST vs GraphQL in one sentence.",
        "providers": ["groq", "openai", "gemini"],
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    for result in body["results"]:
        assert result["status"] == "ok"
        assert result["latency_seconds"] is not None
        assert result["estimated_cost_usd"] is not None
        assert result["quality_score"] is None  # no reference answer given


def test_benchmark_with_reference_answer_scores_quality(client):
    r = client.post("/benchmark", json={
        "prompt": "Explain REST vs GraphQL.",
        "providers": ["groq"],
        "reference_answer": "REST uses fixed endpoints and resources for each type of data.",
    })
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["quality_score"] is not None
    assert 0 <= result["quality_score"] <= 100


def test_benchmark_empty_prompt_returns_400(client):
    r = client.post("/benchmark", json={"prompt": "   "})
    assert r.status_code == 400


def test_benchmark_unknown_provider_returns_400(client):
    r = client.post("/benchmark", json={"prompt": "test", "providers": ["not-a-real-provider"]})
    assert r.status_code == 400


def test_benchmark_defaults_to_available_providers(client):
    r = client.post("/benchmark", json={"prompt": "test prompt"})
    assert r.status_code == 200
    providers_used = {res["provider"] for res in r.json()["results"]}
    assert providers_used == {"groq", "openai", "gemini"}
