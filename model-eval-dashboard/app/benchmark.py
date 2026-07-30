"""
Runs the same prompt across multiple LLM providers and collects comparable
metrics: latency, estimated token usage/cost, and (if a reference answer is
supplied) a simple quality score.
"""
import time

from app.llm_factory import call_provider, MODEL_NAMES, LLMConfigError, LLMCallError
from app.pricing import estimate_cost_usd
from app.tokenizer import estimate_tokens


def _quality_score(output: str, reference_answer: str | None) -> float | None:
    """
    Lightweight, dependency-free quality proxy: percentage of the reference
    answer's distinct words that also appear in the model's output
    (case-insensitive). Returns None if no reference answer was given.

    This is intentionally simple lexical overlap, not semantic similarity --
    chosen to avoid adding an embedding model as a dependency purely for
    scoring. A natural extension (documented in the README) is LLM-as-judge
    scoring or embedding-based cosine similarity for a more nuanced score.
    """
    if not reference_answer or not reference_answer.strip():
        return None
    reference_words = set(reference_answer.lower().split())
    if not reference_words:
        return None
    output_words = set(output.lower().split())
    overlap = len(reference_words & output_words)
    return round(100 * overlap / len(reference_words), 1)


def run_benchmark(
    prompt: str, providers: list[str], reference_answer: str | None = None
) -> list[dict]:
    input_tokens_est = estimate_tokens(prompt)
    results = []

    for provider in providers:
        entry = {"provider": provider, "model": MODEL_NAMES.get(provider, "unknown")}
        start = time.time()
        try:
            output = call_provider(provider, prompt)
            elapsed = time.time() - start
            output_tokens_est = estimate_tokens(output)
            entry.update(
                {
                    "status": "ok",
                    "output": output,
                    "error": None,
                    "latency_seconds": round(elapsed, 2),
                    "input_tokens_est": input_tokens_est,
                    "output_tokens_est": output_tokens_est,
                    "estimated_cost_usd": estimate_cost_usd(
                        provider, input_tokens_est, output_tokens_est
                    ),
                    "quality_score": _quality_score(output, reference_answer),
                }
            )
        except (LLMConfigError, LLMCallError) as exc:
            entry.update(
                {
                    "status": "error",
                    "output": None,
                    "error": str(exc),
                    "latency_seconds": None,
                    "input_tokens_est": input_tokens_est,
                    "output_tokens_est": None,
                    "estimated_cost_usd": None,
                    "quality_score": None,
                }
            )
        results.append(entry)

    return results
