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


class RuntimeSummary(BaseModel):
    total_wall_time: float | None = None
    mean_wall_time: float | None = None
    total_cpu_time: float | None = None
    mean_cpu_time: float | None = None
    timed_target_count: int = 0


class AnalysisReport(BaseModel):
    schema_version: Literal["2"] = "2"
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    by_stratum: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)
    runtime: RuntimeSummary = Field(default_factory=RuntimeSummary)
