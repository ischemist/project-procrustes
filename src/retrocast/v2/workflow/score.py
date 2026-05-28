from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from retrocast.v2.models.candidates import Candidate
from retrocast.v2.models.evaluation import (
    CheckResult,
    CheckStatus,
    ConstraintResult,
    Evaluation,
    ReactionValidity,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.v2.models.route import InChIKeyLevel, Route
from retrocast.v2.models.task import Target, Task, TaskConstraints


class RouteTierChecker(Protocol):
    tier: Tier
    name: str

    def check_route(self, route: Route) -> RouteValidity: ...


class ConstraintChecker(Protocol):
    name: str

    def check_route(self, route: Route, constraints: TaskConstraints) -> ConstraintResult: ...


def score_candidate(
    candidate: Candidate,
    *,
    target: Target,
    constraints: TaskConstraints,
    route_tier_checkers: Sequence[RouteTierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
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

    route = candidate.route
    if route is None:
        raise ValueError("Candidate requires route or failure.")

    _merge_route_validity(validity, _check_route_validity(route, route_tier_checkers))
    constraints_result = constraint_checker.check_route(route, constraints)
    matched_index = _acceptable_match_index(route, target.acceptable_routes, acceptable_match_level)
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
    constraints: TaskConstraints,
    route_tier_checkers: Sequence[RouteTierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> TargetResult:
    scored_candidates = []
    for candidate in candidates:
        scored_candidates.append(
            score_candidate(
                candidate,
                target=target,
                constraints=constraints,
                route_tier_checkers=route_tier_checkers,
                constraint_checker=constraint_checker,
                acceptable_match_level=acceptable_match_level,
            )
        )

    return TargetResult(
        target=target,
        effective_constraints=constraints,
        candidates=scored_candidates,
    )


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    route_tier_checkers: Sequence[RouteTierChecker] = (),
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> Evaluation:
    tiers = [Tier.ZERO, *sorted({checker.tier for checker in route_tier_checkers})]
    return Evaluation(
        task=task,
        tiers=tiers,
        targets={
            target_id: score_target(
                predictions.get(target_id, []),
                target=target,
                constraints=task.constraints.get(target_id, task.default_constraints),
                route_tier_checkers=route_tier_checkers,
                constraint_checker=constraint_checker,
                acceptable_match_level=acceptable_match_level,
            )
            for target_id, target in task.targets.items()
        },
    )


def _tier_zero_validity(candidate: Candidate) -> RouteValidity:
    failure = candidate.failure
    if failure is None:
        return RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)})

    return RouteValidity(
        tiers={
            Tier.ZERO: TierResult(
                status=CheckStatus.FAIL,
                checks=[
                    CheckResult(
                        code=failure.code,
                        status=CheckStatus.FAIL,
                        message=failure.message,
                        details=failure.context,
                    )
                ],
            )
        }
    )


def _check_route_validity(route: Route, route_tier_checkers: Sequence[RouteTierChecker]) -> RouteValidity:
    validity = RouteValidity()
    for checker in route_tier_checkers:
        if checker.tier == Tier.ZERO:
            raise ValueError("Tier.ZERO is reserved for candidate adaptation validity.")
        _merge_route_validity(validity, checker.check_route(route))
    return validity


def _merge_route_validity(validity: RouteValidity, new_validity: RouteValidity) -> None:
    validity.tiers.update(new_validity.tiers)
    reactions_by_id: dict[str, ReactionValidity] = {}
    for reaction in validity.reactions:
        reactions_by_id[reaction.reaction_id] = reaction
    for reaction in new_validity.reactions:
        existing = reactions_by_id.get(reaction.reaction_id)
        if existing is None:
            reactions_by_id[reaction.reaction_id] = reaction
        else:
            existing.tiers.update(reaction.tiers)
    validity.reactions = list(reactions_by_id.values())


def _acceptable_match_index(
    route: Route,
    acceptable_routes: Sequence[Route],
    match_level: InChIKeyLevel,
) -> int | None:
    route_signature = route.signature(match_level)
    for index, acceptable_route in enumerate(acceptable_routes):
        if route_signature == acceptable_route.signature(match_level):
            return index
    return None
