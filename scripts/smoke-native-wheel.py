"""Exercise the installed wheel's direct native API without repository imports."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import tempfile
from importlib.metadata import files, requires
from pathlib import Path

import retrocast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--rdkit-version", required=True)
    parser.add_argument("--allow-python-rdkit", action="store_true")
    args = parser.parse_args()

    if not args.allow_python_rdkit:
        assert importlib.util.find_spec("rdkit") is None
    assert not requires("retrocast")
    distribution_files = {str(path) for path in files("retrocast") or []}
    assert "retrocast/__init__.pyi" in distribution_files
    assert "retrocast/py.typed" in distribution_files
    try:
        importlib.import_module("retrocast.adapters")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("wheel unexpectedly contains the removed Python facade")
    assert retrocast.__engine__ == "rust"
    assert retrocast.engine_info()[1:] == ("RDKit C++", args.rdkit_version)
    assert retrocast.canonicalize_smiles("OCC") == "CCO"

    smiles = retrocast.canonicalize_smiles("CCO")
    target_id = "ethanol"
    task = {
        "name": "wheel-smoke",
        "targets": {
            target_id: {
                "id": target_id,
                "smiles": smiles,
                "inchikey": retrocast.get_inchi_key(smiles),
            }
        },
        "default_constraints": [{"kind": "retrocast.stock_termination", "stock": "test-stock"}],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task_path = root / "task.json.gz"
        raw_path = root / "routes.json.gz"
        manifest_path = root / "manifest.json"
        retrocast.write_task(task, task_path)
        assert retrocast.load_task(task_path)["targets"][target_id]["id"] == target_id
        assert retrocast.resolve_stock_bindings(task) == {target_id: "test-stock"}
        retrocast.write_json_gz({"ethanol": []}, raw_path)
        manifest = retrocast.create_planner_manifest(
            "wheel-smoke",
            "aizynthfinder",
            raw_path,
            [task_path],
            root,
        )
        retrocast.write_json(manifest, manifest_path)
        assert retrocast.verify_planner_manifest(manifest_path, root)["is_valid"]

    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    predictions = retrocast.ingest(
        {target_id: raw},
        "aizynthfinder",
        task,
        workers=2,
    )
    evaluation = retrocast.score(
        predictions,
        task,
        {"test-stock": [retrocast.get_inchi_key("C"), retrocast.get_inchi_key("O")]},
        workers=2,
    )
    report = retrocast.analyze(evaluation, n_boot=10, workers=2)
    candidate = evaluation.to_dict()["targets"][target_id]["candidates"][0]
    assert candidate["constraints"]["status"] == "pass"
    assert report["metrics"]["solv_0[test-stock]_rate"]["value"] == 1.0


if __name__ == "__main__":
    main()
