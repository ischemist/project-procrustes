from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from retrocast.exceptions import UnsupportedValidityTierError
from retrocast.v2.models.candidates import Candidate
from retrocast.v2.models.evaluation import (
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
from retrocast.v2.models.route import InChIKeyLevel, Route
from retrocast.v2.models.task import Target, Task, TaskConstraints


class TierChecker(Protocol):
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
    tier_checkers: Sequence[TierChecker],
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

    route = cast(Route, candidate.route)
    _check_route_validity(route, tier_checkers, validity)
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
    tier_checkers: Sequence[TierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> TargetResult:
    scored_candidates = []
    for candidate in candidates:
        scored_candidate = score_candidate(
            candidate,
            target=target,
            constraints=constraints,
            tier_checkers=tier_checkers,
            constraint_checker=constraint_checker,
            acceptable_match_level=acceptable_match_level,
        )
        scored_candidates.append(scored_candidate)

    return TargetResult(
        target=target,
        effective_constraints=constraints,
        candidates=scored_candidates,
    )


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    tier_checkers: Sequence[TierChecker] = (),
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> Evaluation:
    tiers = [Tier.ZERO, *sorted({checker.tier for checker in tier_checkers})]
    target_results = {}
    for target_id, target in task.targets.items():
        constraints = task.effective_constraints(target_id)
        target_results[target_id] = score_target(
            predictions.get(target_id, []),
            target=target,
            constraints=constraints,
            tier_checkers=tier_checkers,
            constraint_checker=constraint_checker,
            acceptable_match_level=acceptable_match_level,
        )
    return Evaluation(task=task, tiers=tiers, metric_label=task.derived_metric_label(), targets=target_results)


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
    acceptable_routes: Sequence[Route],
    match_level: InChIKeyLevel,
) -> int | None:
    route_signature = route.signature(match_level)
    for index, acceptable_route in enumerate(acceptable_routes):
        if route_signature == acceptable_route.signature(match_level):
            return index
    return None
