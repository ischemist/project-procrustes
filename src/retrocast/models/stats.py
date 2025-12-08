from typing import Literal

from pydantic import BaseModel


class ReliabilityFlag(BaseModel):
    """Warning about the statistical reliability of a result."""

    code: Literal["LOW_N", "EXTREME_P", "OK"]
    message: str


class MetricResult(BaseModel):
    """Result of a single metric estimation (e.g., Top-1 Accuracy)."""

    value: float
    ci_lower: float
    ci_upper: float
    n_samples: int

    reliability: ReliabilityFlag


class StratifiedMetric(BaseModel):
    """Metric broken down by a stratification key (e.g., route_depth)."""

    metric_name: str
    overall: MetricResult
    by_group: dict[int | str, MetricResult]


class ModelStatistics(BaseModel):
    """Complete statistical dump for one model run."""

    model_name: str
    benchmark: str
    stock: str

    # The metrics we care about
    solvability: StratifiedMetric
    top_k_accuracy: dict[int, StratifiedMetric]  # k -> metric

    # Aggregate runtime metrics (in seconds)
    total_wall_time: float | None = None
    total_cpu_time: float | None = None
    mean_wall_time: float | None = None
    mean_cpu_time: float | None = None


class ModelComparison(BaseModel):
    """Statistical comparison between two models (A vs B)."""

    metric: str
    model_a: str
    model_b: str
    diff_mean: float
    diff_ci_lower: float
    diff_ci_upper: float
    is_significant: bool


class RankResult(BaseModel):
    model_name: str
    rank_probs: dict[int, float]
    expected_rank: float
