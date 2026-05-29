from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReliabilityFlag(BaseModel):
    code: Literal["OK", "LOW_N", "EXTREME_P"]
    message: str


class MetricSummary(BaseModel):
    value: float
    count: int
    ci_low: float | None = None
    ci_high: float | None = None
    reliability: ReliabilityFlag | None = None


class AnalysisReport(BaseModel):
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    by_stratum: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)
