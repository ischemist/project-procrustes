"""Python-facing scoring backed by the Rust engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from retrocast.exceptions import UnsupportedValidityTierError
from retrocast.metrics.constraints import TaskConstraintChecker
from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import (
    AcceptableRouteMatch,
    Evaluation,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
)
from retrocast.models.route import InChIKeyLevel, Route
from retrocast.models.task import Target, Task, TaskConstraint
from retrocast.utils.timing import ExecutionStats


class TierChecker(Protocol):
    """Python extension hook for validity tiers not built into RetroCast."""

    tier: Tier
    name: str

    def check_route(self, route: Route) -> RouteValidity: ...


def score_candidate(
    candidate: Candidate,
    *,
    target: Target,
    constraints: Sequence[TaskConstraint],
    tier_checkers: Sequence[TierChecker],
    constraint_checkers: Sequence[TaskConstraintChecker],
    acceptable_match_level: InChIKeyLevel | None = None,
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.PREFIX,
) -> ScoredCandidate:
    """Score one candidate through the same Rust pipeline as a full task."""
    result = score_target(
        [candidate],
        target=target,
        constraints=constraints,
        tier_checkers=tier_checkers,
        constraint_checkers=constraint_checkers,
        acceptable_match_level=acceptable_match_level,
        acceptable_route_match=acceptable_route_match,
    )
    return result.candidates[0]


def score_target(
    candidates: Sequence[Candidate],
    *,
    target: Target,
    constraints: Sequence[TaskConstraint],
    tier_checkers: Sequence[TierChecker],
    constraint_checkers: Sequence[TaskConstraintChecker],
    acceptable_match_level: InChIKeyLevel | None = None,
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.PREFIX,
    wall_time: float | None = None,
    cpu_time: float | None = None,
) -> TargetResult:
    """Score one target through Rust while preserving the focused public API."""
    task = Task(
        name="score-target",
        targets={target.id: target},
        default_constraints=list(constraints),
    )
    stats = None
    if wall_time is not None or cpu_time is not None:
        stats = ExecutionStats(
            wall_time={target.id: wall_time} if wall_time is not None else {},
            cpu_time={target.id: cpu_time} if cpu_time is not None else {},
        )
    evaluation = score(
        {target.id: candidates},
        task,
        tier_checkers=tier_checkers,
        constraint_checkers=constraint_checkers,
        acceptable_match_level=acceptable_match_level,
        acceptable_route_match=acceptable_route_match,
        execution_stats=stats,
    )
    return evaluation.targets[target.id]


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    tier_checkers: Sequence[TierChecker] = (),
    constraint_checkers: Sequence[TaskConstraintChecker] = (),
    acceptable_match_level: InChIKeyLevel | None = None,
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.PREFIX,
    execution_stats: ExecutionStats | None = None,
    workers: int = 1,
) -> Evaluation:
    """Score all task predictions in Rust.

    Custom validity tier checkers remain a Python extension hook. Rust computes
    adaptation validity, constraints, and acceptable-route matches first; the
    hook can only append its explicitly requested tiers.
    """
    from retrocast import native

    evaluation = native.score(
        predictions,
        task,
        constraint_checkers=constraint_checkers,
        acceptable_match_level=acceptable_match_level or InChIKeyLevel.FULL,
        acceptable_route_match=acceptable_route_match,
        execution_stats=execution_stats,
        workers=workers,
    )
    if not tier_checkers:
        return evaluation

    for target_result in evaluation.targets.values():
        for candidate in target_result.candidates:
            if candidate.route is not None:
                _append_custom_tiers(candidate.route, tier_checkers, candidate.validity)
    tiers = [Tier.ZERO, *sorted({checker.tier for checker in tier_checkers})]
    return evaluation.model_copy(update={"tiers": tiers})


def _append_custom_tiers(
    route: Route,
    tier_checkers: Sequence[TierChecker],
    validity: RouteValidity,
) -> None:
    reactions_by_id = {reaction.reaction_id: reaction for reaction in validity.reactions}
    for checker in tier_checkers:
        if checker.tier == Tier.ZERO:
            raise UnsupportedValidityTierError(
                "Tier-0 route validity is reserved for candidate adaptation validity.",
                context={"tier": int(checker.tier), "checker": checker.name},
            )
        result = checker.check_route(route)
        if Tier.ZERO in result.tiers:
            raise UnsupportedValidityTierError(
                "Tier-0 route validity is reserved for candidate adaptation validity.",
                context={"tier": int(Tier.ZERO), "checker": checker.name},
            )
        validity.tiers.update(result.tiers)
        for reaction in result.reactions:
            existing = reactions_by_id.get(reaction.reaction_id)
            if existing is None:
                reactions_by_id[reaction.reaction_id] = reaction
            else:
                existing.tiers.update(reaction.tiers)
    validity.reactions = list(reactions_by_id.values())
