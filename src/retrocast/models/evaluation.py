from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from retrocast.models.chem import Route
from retrocast.models.validity import ConstraintResult, FailureRecord, MetricScope, RouteValidity


class ScoredCandidate(BaseModel):
    rank: int
    route: Route | None = None
    validity: RouteValidity = Field(default_factory=RouteValidity)
    constraint_results: dict[str, ConstraintResult] = Field(default_factory=dict)
    matches_acceptable: bool = False
    matched_acceptable_index: int | None = None
    adapter_failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _require_route_or_failure(self) -> ScoredCandidate:
        if self.route is None and self.adapter_failure is None:
            raise ValueError("ScoredCandidate requires route or adapter_failure.")
        if self.route is not None and self.adapter_failure is not None:
            raise ValueError("ScoredCandidate cannot contain both route and adapter_failure.")
        return self


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
