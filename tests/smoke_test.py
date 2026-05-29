from __future__ import annotations

from importlib.metadata import entry_points


def test_installed_package_runs_minimal_v2_workflow() -> None:
    import retrocast
    from retrocast import Benchmark, Target, TaskConstraints, analyze, get_adapter, ingest_candidates, score
    from retrocast.chem import get_inchi_key
    from retrocast.metrics import TaskConstraintChecker
    from retrocast.typing import InChIKeyStr, SmilesStr

    assert isinstance(retrocast.__version__, str)
    assert entry_points(group="console_scripts")["retrocast"].value == "retrocast.cli.main:main"

    target = Target(
        id="ethanol",
        smiles=SmilesStr("CCO"),
        inchikey=InChIKeyStr(get_inchi_key("CCO")),
    )
    task = Benchmark(
        name="smoke",
        targets={target.id: target},
        default_constraints=TaskConstraints(stock="tiny-stock"),
    )
    raw_payload = {
        "ethanol": {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "smiles": "CCO",
                    "metadata": {"ID": "US123;1", "rsmi": "C.CCO>O.C>CCO"},
                    "children": [
                        {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                        {
                            "type": "mol",
                            "smiles": "CC",
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": "CC",
                                    "metadata": {"ID": "US123;2", "rsmi": "C>C>CC"},
                                    "children": [
                                        {"type": "mol", "smiles": "C", "in_stock": True, "children": []}
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    }

    predictions = ingest_candidates(raw_payload, get_adapter("paroutes"), task)
    evaluation = score(
        predictions,
        task,
        constraint_checker=TaskConstraintChecker(
            stock={InChIKeyStr(get_inchi_key("C"))},
            stock_name="tiny-stock",
        ),
    )
    report = analyze(evaluation, n_boot=10)

    assert len(predictions["ethanol"]) == 1
    assert evaluation.targets["ethanol"].candidates[0].satisfies_solv(0)
    assert report.metrics["solv_0[tiny-stock]_rate"].value == 1.0
