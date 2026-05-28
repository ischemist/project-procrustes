from __future__ import annotations

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models import (
    Candidate,
    CheckStatus,
    FailureRecord,
    Molecule,
    Reaction,
    Route,
    Target,
    Task,
    TaskConstraints,
    Tier,
)
from retrocast.v2.workflow.score import TaskConstraintChecker, TierZeroChecker, score, score_candidate


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


def stock_for(test_route: Route) -> set[InChIKeyStr]:
    return {InChIKeyStr(get_inchi_key(smiles)) for smiles in leaf_smiles(test_route)}


def leaf_smiles(test_route: Route) -> list[str]:
    leaves = []

    def visit(molecule: Molecule) -> None:
        if molecule.product_of is None:
            leaves.append(molecule.smiles)
            return
        for reactant in molecule.product_of.reactants:
            visit(reactant)

    visit(test_route.target)
    return leaves


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
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(),
    )

    assert scored.failed_adaptation()
    assert not scored.satisfies_validity(Tier.ZERO)
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)


def test_valid_route_can_satisfy_tier_zero_and_task() -> None:
    test_route = route()
    scored = score_candidate(
        Candidate(rank=2, route=test_route),
        target=target(),
        constraints=TaskConstraints(stock="stock-a"),
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(stock=stock_for(test_route), stock_name="stock-a"),
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
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(stock=set(), stock_name="stock-a"),
    )

    assert scored.satisfies_validity(Tier.ZERO)
    assert not scored.satisfies_task()
    assert not scored.satisfies_solv(Tier.ZERO)


def test_reaction_validity_is_addressable_by_reaction_id() -> None:
    scored = score_candidate(
        Candidate(rank=1, route=route()),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(),
    )

    result = scored.reaction_tier_result("rc:r:/", Tier.ZERO)
    assert result is not None
    assert result.status == CheckStatus.PASS


def test_empty_reaction_fails_tier_zero() -> None:
    invalid_route = Route(target=molecule("CCO", product_of=Reaction(reactants=[])))

    scored = score_candidate(
        Candidate(rank=1, route=invalid_route),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(),
    )

    assert not scored.satisfies_validity(Tier.ZERO)
    result = scored.reaction_tier_result("rc:r:/", Tier.ZERO)
    assert result is not None
    assert result.status == CheckStatus.FAIL


def test_inchikey_mismatch_fails_tier_zero() -> None:
    invalid_route = Route(target=Molecule(smiles=SmilesStr("CCO"), inchikey=InChIKeyStr(get_inchi_key("C"))))

    scored = score_candidate(
        Candidate(rank=1, route=invalid_route),
        target=target(),
        constraints=TaskConstraints(),
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(),
    )

    assert not scored.satisfies_validity(Tier.ZERO)
    assert scored.tier_result(Tier.ZERO).checks[0].code == "tier0.inchikey_mismatch"


def test_score_preserves_candidate_rank_and_records_acceptable_match() -> None:
    predicted_route = route()
    benchmark_target = target(acceptable_routes=[route(reactant_smiles=("CC", "O")), predicted_route])

    evaluation = score(
        {"ethanol": [Candidate(rank=7, route=predicted_route)]},
        task(benchmark_target),
        tier_checkers=[TierZeroChecker()],
        constraint_checker=TaskConstraintChecker(),
    )

    scored = evaluation.targets["ethanol"].candidates[0]
    assert evaluation.tiers == [Tier.ZERO]
    assert scored.rank == 7
    assert scored.matches_acceptable
    assert scored.matched_acceptable_index == 1
