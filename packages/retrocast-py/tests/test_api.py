from __future__ import annotations

from retrocast.api import analyze_evaluation, ingest_with_adapter, score_predictions
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.metrics import RouteDepthChecker
from retrocast.models import Benchmark, Candidate, CheckStatus, Molecule, Reaction, Route, RouteDepthConstraint, Target
from retrocast.models.evaluation import (
    ConstraintResult,
    Evaluation,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.typing import InChIKeyStr, SmilesStr


def test_score_predictions_applies_non_stock_task_constraints() -> None:
    route = route_for("CCO")
    target = target_for("ethanol", "CCO")
    task = Benchmark(
        name="depth-check",
        targets={target.id: target},
        default_constraints=[RouteDepthConstraint(max_depth=0)],
    )

    evaluation = score_predictions(
        {target.id: [Candidate(rank=1, route=route)]},
        task,
        constraint_checkers=[RouteDepthChecker()],
    )

    scored = evaluation.targets[target.id].candidates[0]
    assert scored.constraints.status == CheckStatus.FAIL
    assert scored.constraints.checks[0].code == "constraint.route_depth.exceeded"


def test_api_ingest_accepts_adapter_names_and_analyze_delegates_to_workflow() -> None:
    target = target_for("ethanol", "CCO")
    task = Benchmark(name="small", targets={target.id: target})
    raw_payload = {target.id: raw_route()}

    collected = ingest_with_adapter(raw_payload, "paroutes", task)
    evaluation = Evaluation(
        task=task,
        tiers=[Tier.ZERO],
        targets={
            target.id: TargetResult(
                target=target,
                effective_constraints=[RouteDepthConstraint(max_depth=1)],
                candidates=[
                    ScoredCandidate(
                        rank=1,
                        route=route_for("CCO"),
                        validity=RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)}),
                        constraints=ConstraintResult(status=CheckStatus.PASS),
                    )
                ],
            )
        },
    )

    report = analyze_evaluation(evaluation, n_boot=4)

    assert target.id in collected
    assert report.metrics["solv_0[task]_rate"].value == 1.0


def raw_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "C.CC>>CCO"},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                    {"type": "mol", "smiles": "CC", "in_stock": True, "children": []},
                ],
            }
        ],
    }


def route_for(smiles: str) -> Route:
    canonical = canonicalize_smiles(smiles)
    return Route(target=molecule(canonical, product_of=Reaction(reactants=[molecule("C")])))


def target_for(target_id: str, smiles: str) -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)))


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)), product_of=product_of)
