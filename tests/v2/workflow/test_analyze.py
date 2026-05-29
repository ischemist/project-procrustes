from __future__ import annotations

from pytest import approx

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models import (
    CheckStatus,
    ConstraintResult,
    Evaluation,
    FailureRecord,
    Molecule,
    Reaction,
    Route,
    RouteValidity,
    ScoredCandidate,
    Target,
    TargetResult,
    Task,
    TaskConstraints,
    Tier,
    TierResult,
)
from retrocast.v2.workflow.analyze import analyze


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        product_of=product_of,
    )


def route() -> Route:
    return Route(target=molecule("CCO", product_of=Reaction(reactants=[molecule("C"), molecule("CO")])))


def route_with_depth(depth: int) -> Route:
    current = molecule("C")
    for _ in range(depth):
        current = molecule("CCO", product_of=Reaction(reactants=[current]))
    return Route(target=current)


def target(target_id: str, *, acceptable_routes: list[Route] | None = None) -> Target:
    canonical = canonicalize_smiles("CCO")
    return Target(
        id=target_id,
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        acceptable_routes=acceptable_routes or [],
    )


def solved_candidate(rank: int, *, matches_acceptable: bool = False) -> ScoredCandidate:
    return ScoredCandidate(
        rank=rank,
        route=route(),
        validity=RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)}),
        constraints=ConstraintResult(status=CheckStatus.PASS),
        matches_acceptable=matches_acceptable,
    )


def failed_candidate(rank: int) -> ScoredCandidate:
    return ScoredCandidate(
        rank=rank,
        failure=FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="ethanol"),
        validity=RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.FAIL)}),
        constraints=ConstraintResult(status=CheckStatus.NOT_EVALUATED),
    )


def unsolved_candidate(rank: int, *, matches_acceptable: bool = False) -> ScoredCandidate:
    return ScoredCandidate(
        rank=rank,
        route=route(),
        validity=RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)}),
        constraints=ConstraintResult(status=CheckStatus.FAIL),
        matches_acceptable=matches_acceptable,
    )


def evaluation(results: dict[str, TargetResult]) -> Evaluation:
    task = Task(name="small", targets={target_id: result.target for target_id, result in results.items()})
    return Evaluation(task=task, tiers=[Tier.ZERO], targets=results)


def result(benchmark_target: Target, candidates: list[ScoredCandidate]) -> TargetResult:
    return TargetResult(target=benchmark_target, effective_constraints=TaskConstraints(), candidates=candidates)


def test_solv_rate_denominator_includes_failed_candidates() -> None:
    report = analyze(
        evaluation(
            {
                "failed": result(target("failed"), [failed_candidate(1)]),
                "solved": result(target("solved"), [solved_candidate(1)]),
            }
        )
    )

    metric = report.metrics["solv_0[task]_rate"]
    assert metric.count == 2
    assert metric.value == approx(0.5)
    assert metric.ci_low is not None
    assert metric.ci_high is not None


def test_mrr_uses_first_solv_satisfying_candidate() -> None:
    report = analyze(
        evaluation(
            {
                "ethanol": result(
                    target("ethanol"),
                    [unsolved_candidate(1), solved_candidate(5), solved_candidate(9)],
                )
            }
        )
    )

    assert report.metrics["mrr_solv_0[task]"].value == approx(0.2)


def test_top_k_reconstruction_is_omitted_without_acceptable_routes() -> None:
    report = analyze(evaluation({"ethanol": result(target("ethanol"), [solved_candidate(1)])}))

    assert "acceptable_reconstruction_top_1[task]" not in report.metrics


def test_top_k_reconstruction_ranks_after_task_satisfaction_filtering() -> None:
    acceptable_route = route()
    report = analyze(
        evaluation(
            {
                "ethanol": result(
                    target("ethanol", acceptable_routes=[acceptable_route]),
                    [
                        unsolved_candidate(1, matches_acceptable=True),
                        solved_candidate(2, matches_acceptable=False),
                        solved_candidate(3, matches_acceptable=True),
                    ],
                )
            }
        ),
        ks=(1, 2),
    )

    assert report.metrics["acceptable_reconstruction_top_1[task]"].value == 0.0
    assert report.metrics["acceptable_reconstruction_top_2[task]"].value == 1.0


def test_analyze_stratifies_by_primary_acceptable_route_depth() -> None:
    report = analyze(
        evaluation(
            {
                "depth-1": result(
                    target("depth-1", acceptable_routes=[route_with_depth(1)]),
                    [solved_candidate(1)],
                ),
                "depth-2": result(
                    target("depth-2", acceptable_routes=[route_with_depth(2)]),
                    [failed_candidate(1)],
                ),
            }
        )
    )

    assert report.by_stratum["depth 1"]["solv_0[task]_rate"].value == 1.0
    assert report.by_stratum["depth 2"]["solv_0[task]_rate"].value == 0.0


def test_analyze_stratifies_by_route_depth_constraint_when_no_acceptable_route_exists() -> None:
    benchmark_target = target("ethanol")
    report = analyze(
        evaluation(
            {
                "ethanol": TargetResult(
                    target=benchmark_target,
                    effective_constraints=TaskConstraints(route_depth=3),
                    candidates=[solved_candidate(1)],
                )
            }
        )
    )

    assert report.by_stratum["depth 3"]["solv_0[task]_rate"].value == 1.0
