from enum import Enum
from pydantic import BaseModel, Field


class SummaryStyle(str, Enum):
    brief = "brief"
    detailed = "detailed"
    bullet_points = "bullet-points"


class SummaryResponse(BaseModel):
    filename: str
    style: SummaryStyle
    provider: str
    model: str
    chunk_count: int
    used_map_reduce: bool
    processing_time_seconds: float
    summary: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str


class ErrorResponse(BaseModel):
    detail: str
