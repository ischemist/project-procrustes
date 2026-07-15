"""Exercise the installed wheel's complete native workflow without repository imports."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import retrocast
from retrocast import _native
from retrocast.adapters import AiZynthFinderAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.metrics.constraints import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker
from retrocast.models import Benchmark, StockTerminationConstraint, Target
from retrocast.typing import InChIKeyStr, SmilesStr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--rdkit-version", required=True)
    parser.add_argument("--allow-python-rdkit", action="store_true")
    args = parser.parse_args()

    if not args.allow_python_rdkit:
        assert importlib.util.find_spec("rdkit") is None
    assert _native.engine_info()[1:] == ("RDKit C++", args.rdkit_version)
    assert canonicalize_smiles("OCC") == "CCO"

    smiles = SmilesStr(canonicalize_smiles("CCO"))
    target = Target(id="ethanol", smiles=smiles, inchikey=InChIKeyStr(get_inchi_key(smiles)))
    task = Benchmark(
        name="wheel-smoke",
        targets={target.id: target},
        default_constraints=[StockTerminationConstraint(stock="test-stock")],
    )
    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    predictions = retrocast.ingest_candidates(
        {target.id: raw},
        AiZynthFinderAdapter(),
        task,
        workers=2,
    )
    evaluation = retrocast.score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(
                stocks={
                    "test-stock": {
                        InChIKeyStr(get_inchi_key("C")),
                        InChIKeyStr(get_inchi_key("O")),
                    }
                }
            ),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
        workers=2,
    )
    report = retrocast.analyze(evaluation, n_boot=10, workers=2)
    assert evaluation.targets[target.id].candidates[0].satisfies_task()
    assert report.metrics["solv_0[test-stock]_rate"].value == 1.0


if __name__ == "__main__":
    main()
