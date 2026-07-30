from pydantic import BaseModel


class BenchmarkRequest(BaseModel):
    prompt: str
    providers: list[str] | None = None  # default: all providers with a configured key
    reference_answer: str | None = None


class BenchmarkResult(BaseModel):
    provider: str
    model: str
    status: str  # "ok" | "error"
    output: str | None = None
    error: str | None = None
    latency_seconds: float | None = None
    input_tokens_est: int | None = None
    output_tokens_est: int | None = None
    estimated_cost_usd: float | None = None
    quality_score: float | None = None  # 0-100, None if no reference_answer given


class BenchmarkResponse(BaseModel):
    prompt: str
    results: list[BenchmarkResult]


class HealthResponse(BaseModel):
    status: str
    available_providers: list[str]
