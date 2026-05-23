from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from retrocast._warnings import DeprecatedFieldAccessMixin
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
            raise ValueError("ScoredCandidate requires either route or adapter_failure.")
        if self.route is not None and self.adapter_failure is not None:
            raise ValueError("ScoredCandidate cannot contain both route and adapter_failure.")
        return self


class ScoredRoute(DeprecatedFieldAccessMixin, BaseModel):
    """
    Legacy route-level compatibility view.

    New evaluation annotations live on ScoredCandidate.
    """

    _deprecated_fields = {
        "is_solved": (
            "ScoredRoute.is_solved",
            'ScoredRoute.constraint_results["stock"]',
            "Historical solved means stock termination, not Solv-N validity.",
        ),
    }

    rank: int
    is_solved: bool  # Legacy alias for stock termination in the stock scope.
    is_stock_terminated: bool | None = None
    matches_acceptable: bool  # Does this route match any acceptable route?
    matched_acceptable_index: int | None = None  # Index of matched acceptable route (if any)

    @model_validator(mode="after")
    def _fill_validity_aliases(self) -> ScoredRoute:
        if self.is_stock_terminated is None:
            self.is_stock_terminated = self.__dict__["is_solved"]
        return self


class TargetEvaluation(DeprecatedFieldAccessMixin, BaseModel):
    """
    The result of evaluating one target against one stock.
    """

    _deprecated_fields = {
        "is_solvable": (
            "TargetEvaluation.is_solvable",
            "TargetEvaluation.has_stock_terminated_route",
            "Historical solvability means stock termination rate, not Solv-N validity.",
        ),
    }

    target_id: str

    # We store ALL scored routes, sorted by rank.
    # This allows O(1) slicing for top-k during stats.
    routes: list[ScoredRoute] = Field(default_factory=list)
    candidates: list[ScoredCandidate] = Field(default_factory=list)

    # Shortcuts for the lazy
    is_solvable: bool = False  # Legacy alias: at least one route is stock-terminated.
    has_stock_terminated_route: bool | None = None
    has_tier_0_valid_route: bool | None = None
    is_solv_0: bool | None = None
    has_tier_1_valid_route: bool | None = None
    is_solv_1: bool | None = None
    acceptable_rank: int | None = None  # Rank of first solved acceptable match (None if not found)
    tier_validity_ranks: dict[int, int | None] = Field(default_factory=dict)
    solv_ranks: dict[str, dict[int, int | None]] = Field(default_factory=dict)
    top_k_ranks: dict[str, int | None] = Field(default_factory=dict)

    # Properties used for stratification (route length, convergence)
    stratification_length: int | None = None
    stratification_is_convergent: bool | None = None

    # Runtime metrics for this target (in seconds)
    wall_time: float | None = None
    cpu_time: float | None = None

    @model_validator(mode="after")
    def _fill_target_validity_aliases(self) -> TargetEvaluation:
        is_solvable = self.__dict__["is_solvable"]
        if self.has_stock_terminated_route is None:
            self.has_stock_terminated_route = is_solvable
        elif "is_solvable" not in self.model_fields_set:
            self.is_solvable = self.has_stock_terminated_route
        elif is_solvable != self.has_stock_terminated_route:
            raise ValueError("is_solvable and has_stock_terminated_route disagree.")
        return self


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
