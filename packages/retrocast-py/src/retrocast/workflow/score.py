from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from retrocast.exceptions import UnsupportedValidityTierError
from retrocast.metrics.constraints import TaskConstraintChecker, check_task_constraints
from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import (
    AcceptableRouteMatch,
    CheckResult,
    CheckStatus,
    ConstraintResult,
    Evaluation,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.models.route import InChIKeyLevel, Route
from retrocast.models.task import Target, Task, TaskConstraint
from retrocast.utils.timing import ExecutionStats


class TierChecker(Protocol):
    tier: Tier
    name: str

    def check_route(self, route: Route) -> RouteValidity: ...


@dataclass(frozen=True, slots=True)
class AcceptableRouteIdentity:
    index: int
    depth: int
    signature: str


def score_candidate(
    candidate: Candidate,
    *,
    target: Target,
    constraints: Sequence[TaskConstraint],
    tier_checkers: Sequence[TierChecker],
    constraint_checkers: Sequence[TaskConstraintChecker],
    acceptable_match_level: InChIKeyLevel | None = None,
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.PREFIX,
    _acceptable_identities: Sequence[AcceptableRouteIdentity] | None = None,
) -> ScoredCandidate:
    """Score one candidate while preserving failed adaptation slots."""
    validity = _tier_zero_validity(candidate)
    if candidate.failure is not None:
        return ScoredCandidate(
            rank=candidate.rank,
            failure=candidate.failure,
            validity=validity,
            constraints=ConstraintResult(status=CheckStatus.NOT_EVALUATED),
        )

    route = cast(Route, candidate.route)
    _check_route_validity(route, tier_checkers, validity)
    constraints_result = check_task_constraints(route, constraints, constraint_checkers)
    route_match_level = acceptable_match_level or InChIKeyLevel.FULL
    acceptable_identities = (
        _acceptable_identities
        if _acceptable_identities is not None
        else _acceptable_route_identities(target.acceptable_routes, route_match_level)
    )
    matched_index = _acceptable_match_index(route, acceptable_identities, route_match_level, acceptable_route_match)
    return ScoredCandidate(
        rank=candidate.rank,
        route=route,
        validity=validity,
        constraints=constraints_result,
        matches_acceptable=matched_index is not None,
        matched_acceptable_index=matched_index,
    )


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
    scored_candidates = []
    route_match_level = acceptable_match_level or InChIKeyLevel.FULL
    acceptable_identities = _acceptable_route_identities(target.acceptable_routes, route_match_level)
    for candidate in candidates:
        scored_candidate = score_candidate(
            candidate,
            target=target,
            constraints=constraints,
            tier_checkers=tier_checkers,
            constraint_checkers=constraint_checkers,
            acceptable_match_level=route_match_level,
            acceptable_route_match=acceptable_route_match,
            _acceptable_identities=acceptable_identities,
        )
        scored_candidates.append(scored_candidate)

    return TargetResult(
        target=target,
        effective_constraints=list(constraints),
        candidates=scored_candidates,
        wall_time=wall_time,
        cpu_time=cpu_time,
    )


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    tier_checkers: Sequence[TierChecker] = (),
    constraint_checkers: Sequence[TaskConstraintChecker] = (),
    acceptable_match_level: InChIKeyLevel | None = None,
    acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.PREFIX,
    execution_stats: ExecutionStats | None = None,
) -> Evaluation:
    tiers = [Tier.ZERO, *sorted({checker.tier for checker in tier_checkers})]
    target_results = {}
    route_match_level = acceptable_match_level or InChIKeyLevel.FULL
    for target_id, target in task.targets.items():
        constraints = task.effective_constraints(target_id)
        target_results[target_id] = score_target(
            predictions.get(target_id, []),
            target=target,
            constraints=constraints,
            tier_checkers=tier_checkers,
            constraint_checkers=constraint_checkers,
            acceptable_match_level=route_match_level,
            acceptable_route_match=acceptable_route_match,
            wall_time=execution_stats.wall_time.get(target_id) if execution_stats is not None else None,
            cpu_time=execution_stats.cpu_time.get(target_id) if execution_stats is not None else None,
        )
    return Evaluation(
        task=task,
        tiers=tiers,
        metric_label=task.derived_metric_label(),
        acceptable_match_level=route_match_level,
        acceptable_route_match=acceptable_route_match,
        targets=target_results,
    )


def _tier_zero_validity(candidate: Candidate) -> RouteValidity:
    failure = candidate.failure
    if failure is None:
        return RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)})

    failure_check = CheckResult(
        code=failure.code,
        status=CheckStatus.FAIL,
        message=failure.message,
        details=failure.context,
    )
    tier_result = TierResult(status=CheckStatus.FAIL, checks=[failure_check])
    return RouteValidity(tiers={Tier.ZERO: tier_result})


def _check_route_validity(route: Route, tier_checkers: Sequence[TierChecker], validity: RouteValidity) -> None:
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
                # Multiple tier checkers can report validity for the same route reaction.
                existing.tiers.update(reaction.tiers)
    validity.reactions = list(reactions_by_id.values())


def _acceptable_match_index(
    route: Route,
    acceptable_identities: Sequence[AcceptableRouteIdentity],
    match_level: InChIKeyLevel,
    route_match: AcceptableRouteMatch,
) -> int | None:
    if route_match == AcceptableRouteMatch.EXACT:
        route_signature = route.signature(match_level)
        for identity in acceptable_identities:
            if route_signature == identity.signature:
                return identity.index
        return None

    if route_match == AcceptableRouteMatch.PREFIX:
        route_depth = route.depth()
        route_signatures_by_depth: dict[int, str] = {}
        best: AcceptableRouteIdentity | None = None
        for identity in acceptable_identities:
            if route_depth < identity.depth:
                continue
            if identity.depth not in route_signatures_by_depth:
                route_signatures_by_depth[identity.depth] = route.signature(match_level, depth=identity.depth)
            route_signature = route_signatures_by_depth[identity.depth]
            if route_signature == identity.signature and (best is None or identity.depth >= best.depth):
                best = identity
        return best.index if best is not None else None

    raise ValueError(f"unsupported acceptable route match mode: {route_match}")


def _acceptable_route_identities(
    acceptable_routes: Sequence[Route],
    match_level: InChIKeyLevel,
) -> tuple[AcceptableRouteIdentity, ...]:
    return tuple(
        AcceptableRouteIdentity(index=index, depth=route.depth(), signature=route.signature(match_level))
        for index, route in enumerate(acceptable_routes)
    )
