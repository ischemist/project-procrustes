from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.io.blob import save_json_gz
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.cli.main import main
from retrocast.v2.io import (
    load_analysis_report,
    load_candidates,
    load_collected_candidates,
    load_evaluation,
    save_benchmark,
)
from retrocast.v2.models import Benchmark, Target, TaskConstraints


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
                    {
                        "type": "mol",
                        "smiles": "CC",
                        "children": [
                            {
                                "type": "reaction",
                                "smiles": "CC",
                                "metadata": {"ID": "US123;2", "rsmi": "C>>CC"},
                                "children": [{"type": "mol", "smiles": "C", "in_stock": True, "children": []}],
                            }
                        ],
                    },
                ],
            }
        ],
    }


def invalid_raw_route() -> dict:
    route = raw_route()
    route["children"][0]["children"][1]["smiles"] = "not-smiles"
    return route


def benchmark() -> Benchmark:
    smiles = canonicalize_smiles("CCO")
    target = Target(id="ethanol", smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
    return Benchmark(
        name="small",
        targets={target.id: target},
        default_constraints=TaskConstraints(stock="test-stock"),
    )


def write_stock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "InChIKey"])
        writer.writerow(["C", get_inchi_key("C")])


def run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["retrocast", *args])
    main()


def test_v2_adapt_cli_writes_candidates_and_manifest(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "raw.json.gz"
    output_path = tmp_path / "candidates.json.gz"
    save_json_gz({"ok": raw_route(), "bad": invalid_raw_route()}, raw_path)

    run_cli(
        monkeypatch,
        "adapt",
        "--input",
        str(raw_path),
        "--output",
        str(output_path),
        "--adapter",
        "paroutes",
    )

    candidates = load_candidates(output_path)
    assert len(candidates) == 2
    assert any(candidate.failure is not None for candidate in candidates)
    manifest = json.loads((tmp_path / "candidates.manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2"
    assert manifest["statistics"]["failures_by_code"]


def test_v2_project_cli_ingest_score_analyze(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "2-raw" / "test-model" / "small"
    raw_dir.mkdir(parents=True)
    (data_dir / "1-benchmarks" / "definitions").mkdir(parents=True)
    write_stock(data_dir / "1-benchmarks" / "stocks" / "test-stock.csv.gz")

    save_benchmark(benchmark(), data_dir / "1-benchmarks" / "definitions" / "small.json.gz")
    save_json_gz({"ethanol": raw_route()}, raw_dir / "results.json.gz")
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "2", "directives": {"adapter": "paroutes", "raw_results_filename": "results.json.gz"}}
        ),
        encoding="utf-8",
    )

    run_cli(monkeypatch, "--data-dir", str(data_dir), "ingest", "--model", "test-model", "--dataset", "small")
    candidates_path = data_dir / "3-processed" / "small" / "test-model" / "candidates.json.gz"
    assert len(load_collected_candidates(candidates_path)["ethanol"]) == 1
    assert (candidates_path.parent / "manifest.json").exists()

    run_cli(monkeypatch, "--data-dir", str(data_dir), "score", "--model", "test-model", "--dataset", "small")
    evaluation_path = data_dir / "4-scored" / "small" / "test-model" / "test-stock" / "evaluation.json.gz"
    evaluation = load_evaluation(evaluation_path)
    assert evaluation.schema_version == "2"
    assert evaluation.targets["ethanol"].candidates[0].satisfies_task()

    run_cli(
        monkeypatch,
        "--data-dir",
        str(data_dir),
        "analyze",
        "--model",
        "test-model",
        "--dataset",
        "small",
        "--n-boot",
        "10",
    )
    analysis_path = data_dir / "5-results" / "small" / "test-model" / "test-stock" / "analysis.json.gz"
    report = load_analysis_report(analysis_path)
    assert report.metrics["solv_0[test-stock]_rate"].value == 1.0
    assert (analysis_path.parent / "report.md").exists()
    assert (analysis_path.parent / "manifest.json").exists()
