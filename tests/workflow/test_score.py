from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import UnsupportedValidityTierError
from retrocast.models import (
    Candidate,
    CheckStatus,
    ConstraintResult,
    Evaluation,
    FailureRecord,
    Molecule,
    Reaction,
    ReactionValidity,
    Route,
    RouteValidity,
    Target,
    Task,
    TaskConstraints,
    Tier,
    TierResult,
)
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.utils.timing import ExecutionStats
from retrocast.workflow.score import AcceptableRouteMatch, score, score_candidate


class FixedTierChecker:
    tier = Tier.ONE
    name = "fixed-tier-one"

    def __init__(self, status: CheckStatus = CheckStatus.PASS) -> None:
        self.status = status

    def check_route(self, route: Route) -> RouteValidity:
        reaction_tiers = {}
        try:
            reaction_id = route.reaction_at("rc:r:/").id()
        except KeyError:
            reactions = []
        else:
            reaction_tiers[self.tier] = TierResult(status=self.status)
            reactions = [ReactionValidity(reaction_id=reaction_id, tiers=reaction_tiers)]
        return RouteValidity(tiers={self.tier: TierResult(status=self.status)}, reactions=reactions)


class InvalidTierZeroRouteChecker(FixedTierChecker):
    tier = Tier.ZERO
    name = "invalid-tier-zero"


class InvalidReturnedTierZeroRouteChecker(FixedTierChecker):
    name = "invalid-returned-tier-zero"

    def check_route(self, route: Route) -> RouteValidity:
        return RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.FAIL)})


class FixedTierTwoChecker(FixedTierChecker):
    tier = Tier.TWO


class FixedConstraintChecker:
    name = "fixed-task"

    def __init__(self, status: CheckStatus = CheckStatus.PASS) -> None:
        self.status = status

    def check_route(self, route: Route, constraints: TaskConstraints) -> ConstraintResult:
        return ConstraintResult(status=self.status)


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        product_of=product_of,
    )


def route(target_smiles: str = "CCO", reactant_smiles: tuple[str, ...] = ("C", "CO")) -> Route:
    return Route(
        target=molecule(
            target_smiles,
            product_of=Reaction(reactants=[molecule(smiles) for smiles in reactant_smiles]),
        )
    )


def target(acceptable_routes: list[Route] | None = None) -> Target:
    canonical = canonicalize_smiles("CCO")
    return Target(
        id="ethanol",
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        acceptable_routes=acceptable_routes or [],
    )


def task(benchmark_target: Target) -> Task:
    return Task(name="small", targets={benchmark_target.id: benchmark_target})


def test_failed_candidate_does_not_satisfy_solv_zero() -> None:
    candidate = Candidate(
        rank=1,
        failure=FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="ethanol"),
    )

    scored = score_candidate(
        candidate,
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert scored.failed_adaptation()
    assert not scored.satisfies_validity(Tier.ZERO)
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)


def test_valid_route_satisfies_tier_zero_from_candidate_boundary() -> None:
    scored = score_candidate(
        Candidate(rank=2, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert scored.has_route()
    assert scored.satisfies_validity(Tier.ZERO)
    assert scored.satisfies_task()
    assert scored.satisfies_solv(Tier.ZERO)


def test_task_constraint_failure_prevents_solv_zero() -> None:
    scored = score_candidate(
        Candidate(rank=1, route=route()),
        target=target(),
        constraints=TaskConstraints(stock="stock-a"),
        tier_checkers=[FixedTierChecker()],
        constraint_checker=FixedConstraintChecker(CheckStatus.FAIL),
    )

    assert scored.satisfies_validity(Tier.ZERO)
    assert scored.satisfies_validity(Tier.ONE)
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)
    assert not scored.satisfies_solv(Tier.ONE)


def test_reaction_validity_is_addressable_by_reaction_id() -> None:
    scored = score_candidate(
        Candidate(rank=1, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[FixedTierChecker()],
        constraint_checker=FixedConstraintChecker(),
    )

    result = scored.reaction_tier_result("rc:r:/", Tier.ONE)
    assert result is not None
    assert result.status == CheckStatus.PASS


def test_route_tier_checker_cannot_claim_tier_zero() -> None:
    with pytest.raises(UnsupportedValidityTierError) as exc_info:
        score_candidate(
            Candidate(rank=1, route=route()),
            target=target(),
            constraints=TaskConstraints(),
            tier_checkers=[InvalidTierZeroRouteChecker()],
            constraint_checker=FixedConstraintChecker(),
        )
    assert exc_info.value.code == "validity.unsupported_tier"
    assert exc_info.value.context == {"tier": 0, "checker": "invalid-tier-zero"}


def test_route_tier_checker_cannot_return_tier_zero() -> None:
    with pytest.raises(UnsupportedValidityTierError) as exc_info:
        score_candidate(
            Candidate(rank=1, route=route()),
            target=target(),
            constraints=TaskConstraints(),
            tier_checkers=[InvalidReturnedTierZeroRouteChecker()],
            constraint_checker=FixedConstraintChecker(),
        )
    assert exc_info.value.code == "validity.unsupported_tier"
    assert exc_info.value.context == {"tier": 0, "checker": "invalid-returned-tier-zero"}


def test_reaction_validity_merges_results_from_multiple_tier_checkers() -> None:
    scored = score_candidate(
        Candidate(rank=1, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[FixedTierChecker(), FixedTierTwoChecker(CheckStatus.FAIL)],
        constraint_checker=FixedConstraintChecker(),
    )

    tier_one = scored.reaction_tier_result("rc:r:/", Tier.ONE)
    tier_two = scored.reaction_tier_result("rc:r:/", Tier.TWO)
    assert tier_one is not None
    assert tier_one.status == CheckStatus.PASS
    assert tier_two is not None
    assert tier_two.status == CheckStatus.FAIL


def test_score_preserves_candidate_rank_and_records_acceptable_match() -> None:
    predicted_route = route()
    benchmark_target = target(acceptable_routes=[route(reactant_smiles=("CC", "O")), predicted_route])

    evaluation = score(
        {"ethanol": [Candidate(rank=7, route=predicted_route)]},
        task(benchmark_target),
        tier_checkers=[FixedTierChecker()],
        constraint_checker=FixedConstraintChecker(),
    )

    scored = evaluation.targets["ethanol"].candidates[0]
    assert evaluation.tiers == [Tier.ZERO, Tier.ONE]
    assert evaluation.acceptable_route_match == AcceptableRouteMatch.PREFIX
    assert scored.rank == 7
    assert scored.matches_acceptable
    assert scored.matched_acceptable_index == 1


def test_score_records_acceptable_match_when_reference_is_prediction_prefix() -> None:
    acceptable_route = route(reactant_smiles=("C", "CO"))
    predicted_route = Route(
        target=molecule(
            "CCO",
            product_of=Reaction(
                reactants=[
                    molecule("C", product_of=Reaction(reactants=[molecule("N")])),
                    molecule("CO"),
                ]
            ),
        )
    )
    benchmark_target = target(acceptable_routes=[acceptable_route])

    scored = score_candidate(
        Candidate(rank=1, route=predicted_route),
        target=benchmark_target,
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert predicted_route.signature() != acceptable_route.signature()
    assert scored.matches_acceptable
    assert scored.matched_acceptable_index == 0


def test_score_prefers_deepest_acceptable_prefix_match() -> None:
    shallow_route = route(reactant_smiles=("C", "CO"))
    deep_route = Route(
        target=molecule(
            "CCO",
            product_of=Reaction(
                reactants=[
                    molecule("C", product_of=Reaction(reactants=[molecule("N")])),
                    molecule("CO"),
                ]
            ),
        )
    )
    benchmark_target = target(acceptable_routes=[shallow_route, deep_route])

    scored = score_candidate(
        Candidate(rank=1, route=deep_route),
        target=benchmark_target,
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert scored.matches_acceptable
    assert scored.matched_acceptable_index == 1


def test_score_can_require_exact_acceptable_route_match() -> None:
    acceptable_route = route(reactant_smiles=("C", "CO"))
    predicted_route = Route(
        target=molecule(
            "CCO",
            product_of=Reaction(
                reactants=[
                    molecule("C", product_of=Reaction(reactants=[molecule("N")])),
                    molecule("CO"),
                ]
            ),
        )
    )
    benchmark_target = target(acceptable_routes=[acceptable_route])

    scored = score_candidate(
        Candidate(rank=1, route=predicted_route),
        target=benchmark_target,
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
        acceptable_route_match=AcceptableRouteMatch.EXACT,
    )

    assert not scored.matches_acceptable
    assert scored.matched_acceptable_index is None


def test_legacy_evaluation_defaults_to_exact_acceptable_route_match() -> None:
    benchmark_target = target()

    evaluation = Evaluation(task=task(benchmark_target))

    assert evaluation.acceptable_route_match == AcceptableRouteMatch.EXACT


def test_score_does_not_match_candidate_shorter_than_reference_route() -> None:
    acceptable_route = Route(
        target=molecule(
            "CCO",
            product_of=Reaction(
                reactants=[
                    molecule("C", product_of=Reaction(reactants=[molecule("N")])),
                    molecule("CO"),
                ]
            ),
        )
    )
    predicted_route = route(reactant_smiles=("C", "CO"))
    benchmark_target = target(acceptable_routes=[acceptable_route])

    scored = score_candidate(
        Candidate(rank=1, route=predicted_route),
        target=benchmark_target,
        constraints=TaskConstraints(),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert predicted_route.signature(depth=predicted_route.depth()) != acceptable_route.signature()
    assert not scored.matches_acceptable
    assert scored.matched_acceptable_index is None


def test_score_records_metric_label_from_task() -> None:
    benchmark_target = target()
    benchmark_task = Task(
        name="scoped",
        targets={benchmark_target.id: benchmark_target},
        default_constraints=TaskConstraints(stock="buyables", route_depth=3),
    )

    evaluation = score(
        {"ethanol": [Candidate(rank=1, route=route())]},
        benchmark_task,
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
    )

    assert evaluation.metric_label == "buyables+depth"


def test_score_carries_execution_stats_to_target_results() -> None:
    benchmark_target = target()
    execution_stats = ExecutionStats(wall_time={"ethanol": 12.5}, cpu_time={"ethanol": 3.25})

    evaluation = score(
        {"ethanol": [Candidate(rank=1, route=route())]},
        task(benchmark_target),
        tier_checkers=[],
        constraint_checker=FixedConstraintChecker(),
        execution_stats=execution_stats,
    )

    target_result = evaluation.targets["ethanol"]
    assert target_result.wall_time == 12.5
    assert target_result.cpu_time == 3.25
