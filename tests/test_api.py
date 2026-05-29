from __future__ import annotations

from retrocast.api import score_predictions
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.models import Benchmark, Candidate, CheckStatus, Molecule, Reaction, Route, Target, TaskConstraints
from retrocast.typing import InChIKeyStr, SmilesStr


def test_score_predictions_applies_non_stock_task_constraints() -> None:
    route = route_for("CCO")
    target = target_for("ethanol", "CCO")
    task = Benchmark(
        name="depth-check",
        targets={target.id: target},
        default_constraints=TaskConstraints(route_depth=0),
    )

    evaluation = score_predictions({target.id: [Candidate(rank=1, route=route)]}, task)

    scored = evaluation.targets[target.id].candidates[0]
    assert scored.constraints.status == CheckStatus.FAIL
    assert scored.constraints.checks[0].code == "constraint.route_depth.exceeded"


def route_for(smiles: str) -> Route:
    canonical = canonicalize_smiles(smiles)
    return Route(target=molecule(canonical, product_of=Reaction(reactants=[molecule("C")])))


def target_for(target_id: str, smiles: str) -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)))


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)), product_of=product_of)
