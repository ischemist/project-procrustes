from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from retrocast.models.evaluation import Tier

TierInput = Tier | int


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
    bootstrap_resamples: int | None = None
    runtime: RuntimeSummary = Field(default_factory=RuntimeSummary)

    def validity_rate(self, tier: TierInput) -> MetricSummary | None:
        return self.metrics.get(f"tier_{_tier_value(tier)}_validity_rate")

    def mrr_validity(self, tier: TierInput) -> MetricSummary | None:
        return self.metrics.get(f"tier_{_tier_value(tier)}_validity_mrr")

    def solv_rate(self, tier: TierInput, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"solv_{_tier_value(tier)}[{label}]_rate")

    def mrr_solv(self, tier: TierInput, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"solv_{_tier_value(tier)}[{label}]_mrr")

    def reconstruction(self, top_k: int, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"acceptable_reconstruction_top_{_positive_int(top_k, 'top_k')}[{label}]")

    def root_reconstruction(self, top_k: int, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"acceptable_root_reconstruction_top_{_positive_int(top_k, 'top_k')}[{label}]")

    def reconstruction_given_root(self, top_k: int, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"acceptable_reconstruction_given_root_top_{_positive_int(top_k, 'top_k')}[{label}]")

    def prefix_reconstruction(self, depth: int, top_k: int, *, label: str = "task") -> MetricSummary | None:
        resolved_depth = _positive_int(depth, "depth")
        resolved_top_k = _positive_int(top_k, "top_k")
        return self.metrics.get(
            f"acceptable_prefix_reconstruction_depth_{resolved_depth}_top_{resolved_top_k}[{label}]"
        )

    def distinct_root_reactions(self, top_k: int, *, label: str = "task") -> MetricSummary | None:
        return self.metrics.get(f"distinct_root_reactions_top_{_positive_int(top_k, 'top_k')}[{label}]")


def _tier_value(tier: TierInput) -> int:
    return int(Tier(tier))


def _positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value
