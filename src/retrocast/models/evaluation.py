from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator

from retrocast.models.candidates import FailureRecord
from retrocast.models.route import InChIKeyLevel, ReactionId, Route
from retrocast.models.task import Target, Task, TaskConstraint, hydrate_task_constraints


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class Tier(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class AcceptableRouteMatch(StrEnum):
    PREFIX = "prefix"
    EXACT = "exact"


class CheckResult(BaseModel):
    code: str
    status: CheckStatus = CheckStatus.FAIL
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TierResult(BaseModel):
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ReactionValidity(BaseModel):
    reaction_id: ReactionId
    tiers: dict[Tier, TierResult] = Field(default_factory=dict)


class RouteValidity(BaseModel):
    tiers: dict[Tier, TierResult] = Field(default_factory=dict)
    reactions: list[ReactionValidity] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ScoredCandidate(BaseModel):
    rank: int = Field(ge=1)
    route: Route | None = None
    failure: FailureRecord | None = None
    validity: RouteValidity = Field(default_factory=RouteValidity)
    constraints: ConstraintResult = Field(default_factory=lambda: ConstraintResult(status=CheckStatus.NOT_EVALUATED))
    matches_acceptable: bool = False
    matched_acceptable_index: int | None = None

    @model_validator(mode="after")
    def _require_route_or_failure(self) -> ScoredCandidate:
        if self.route is None and self.failure is None:
            raise ValueError("ScoredCandidate requires route or failure.")
        if self.route is not None and self.failure is not None:
            raise ValueError("ScoredCandidate cannot contain both route and failure.")
        return self

    def has_route(self) -> bool:
        return self.route is not None

    def failed_adaptation(self) -> bool:
        return self.failure is not None

    def tier_result(self, tier: Tier | int) -> TierResult:
        return self.validity.tiers.get(Tier(tier), TierResult(status=CheckStatus.NOT_EVALUATED))

    def reaction_tier_result(self, reaction_id: ReactionId, tier: Tier | int) -> TierResult | None:
        for reaction in self.validity.reactions:
            if reaction.reaction_id == reaction_id:
                return reaction.tiers.get(Tier(tier))
        return None

    def satisfies_validity(self, tier: Tier | int) -> bool:
        return self.tier_result(tier).status == CheckStatus.PASS

    def satisfies_task(self) -> bool:
        return self.constraints.status == CheckStatus.PASS

    def satisfies_solv(self, tier: Tier | int) -> bool:
        return self.satisfies_validity(tier) and self.satisfies_task()


class TargetResult(BaseModel):
    target: Target
    # Pydantic serializes fields through their annotation. Without SerializeAsAny,
    # subclass payload fields like stock/smiles/max_depth are dropped.
    effective_constraints: list[SerializeAsAny[TaskConstraint]]
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    wall_time: float | None = None
    cpu_time: float | None = None

    @field_validator("effective_constraints", mode="before")
    @classmethod
    def _parse_effective_constraints(cls, value: object) -> list[TaskConstraint]:
        return hydrate_task_constraints(value)


class Evaluation(BaseModel):
    task: Task
    tiers: list[Tier] = Field(default_factory=list)
    metric_label: str = "task"
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.EXACT
    targets: dict[str, TargetResult] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"
