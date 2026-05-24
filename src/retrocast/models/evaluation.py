from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from retrocast.models.chem import Route
from retrocast.models.validity import ConstraintResult, FailureRecord, MetricScope, RouteValidity, ScopeId, ValidityTier


def tier_rank_key(tier: ValidityTier) -> str:
    return f"tier {tier}"


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

    def satisfies_validity(self, tier: ValidityTier = 0) -> bool:
        return self.validity.satisfies_validity(tier=tier)

    def satisfies_constraints(self, scope: ScopeId = "stock") -> bool:
        constraint_result = self.constraint_results.get(scope)
        return constraint_result is not None and constraint_result.status == "pass"

    def satisfies_solv(self, tier: ValidityTier = 0, scope: ScopeId = "stock") -> bool:
        return self.satisfies_validity(tier=tier) and self.satisfies_constraints(scope=scope)


class TargetEvaluation(BaseModel):
    """
    The result of evaluating one target against one stock.
    """

    target_id: str

    candidates: list[ScoredCandidate] = Field(default_factory=list)

    has_stock_terminated_route: bool = False
    first_valid_ranks: dict[str, int | None] = Field(default_factory=dict)
    first_solv_ranks: dict[ScopeId, dict[str, int | None]] = Field(default_factory=dict)
    first_reconstruction_ranks: dict[ScopeId, int | None] = Field(default_factory=dict)

    # Properties used for stratification (route length, convergence)
    stratification_length: int | None = None
    stratification_is_convergent: bool | None = None

    # Runtime metrics for this target (in seconds)
    wall_time: float | None = None
    cpu_time: float | None = None

    def first_valid_rank(self, tier: ValidityTier = 0) -> int | None:
        return self.first_valid_ranks.get(tier_rank_key(tier))

    def first_solv_rank(self, tier: ValidityTier = 0, scope: ScopeId = "stock") -> int | None:
        return self.first_solv_ranks.get(scope, {}).get(tier_rank_key(tier))

    def reconstruction_rank(self, scope: ScopeId = "stock") -> int | None:
        return self.first_reconstruction_ranks.get(scope)


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
