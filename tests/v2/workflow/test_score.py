from __future__ import annotations

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models import (
    Candidate,
    CheckStatus,
    ConstraintResult,
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
from retrocast.v2.workflow.score import score, score_candidate


class FixedTierChecker:
    tier = Tier.ZERO
    name = "fixed-tier-zero"

    def __init__(self, status: CheckStatus = CheckStatus.PASS) -> None:
        self.status = status

    def check_route(self, route: Route) -> RouteValidity:
        reaction_tiers = {}
        try:
            reaction_id = route.reaction_at("rc:r:/").id()
        except KeyError:
            reactions = []
        else:
            reaction_tiers[Tier.ZERO] = TierResult(status=self.status)
            reactions = [ReactionValidity(reaction_id=reaction_id, tiers=reaction_tiers)]
        return RouteValidity(tiers={Tier.ZERO: TierResult(status=self.status)}, reactions=reactions)


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
        tier_checkers=[FixedTierChecker()],
        constraint_checker=FixedConstraintChecker(),
    )

    assert scored.failed_adaptation()
    assert not scored.satisfies_validity(Tier.ZERO)
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)


def test_valid_route_can_satisfy_tier_zero_and_task() -> None:
    scored = score_candidate(
        Candidate(rank=2, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[FixedTierChecker()],
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
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)


def test_reaction_validity_is_addressable_by_reaction_id() -> None:
    scored = score_candidate(
        Candidate(rank=1, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[FixedTierChecker()],
        constraint_checker=FixedConstraintChecker(),
    )

    result = scored.reaction_tier_result("rc:r:/", Tier.ZERO)
    assert result is not None
    assert result.status == CheckStatus.PASS


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
    assert evaluation.tiers == [Tier.ZERO]
    assert scored.rank == 7
    assert scored.matches_acceptable
    assert scored.matched_acceptable_index == 1
