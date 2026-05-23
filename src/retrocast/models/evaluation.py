from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from retrocast.models.chem import Route
from retrocast.models.validity import ConstraintResult, MetricScope, RouteValidity


class ScoredCandidate(BaseModel):
    rank: int
    route: Route
    validity: RouteValidity = Field(default_factory=RouteValidity)
    constraint_results: dict[str, ConstraintResult] = Field(default_factory=dict)
    matches_acceptable: bool = False
    matched_acceptable_index: int | None = None


class TargetEvaluation(BaseModel):
    """
    The result of evaluating one target against one stock.
    """

    target_id: str

    candidates: list[ScoredCandidate] = Field(default_factory=list)

    has_stock_terminated_route: bool = False
    has_tier_0_valid_route: bool | None = None
    is_solv_0: bool | None = None
    acceptable_rank: int | None = None
    tier_validity_ranks: dict[int, int | None] = Field(default_factory=dict)
    solv_ranks: dict[str, dict[int, int | None]] = Field(default_factory=dict)
    top_k_ranks: dict[str, int | None] = Field(default_factory=dict)

    # Properties used for stratification (route length, convergence)
    stratification_length: int | None = None
    stratification_is_convergent: bool | None = None

    # Runtime metrics for this target (in seconds)
    wall_time: float | None = None
    cpu_time: float | None = None


class EvaluationResults(BaseModel):
    """
    The complete dump of a scoring run.
    """

    model_name: str
    benchmark_name: str
    stock_name: str
    has_acceptable_routes: bool  # Whether the benchmark has acceptable routes (not whether model found them)
    metric_scopes: list[MetricScope] = Field(default_factory=list)

    # Map target_id -> Evaluation
    results: dict[str, TargetEvaluation] = Field(default_factory=dict)

    # Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)
