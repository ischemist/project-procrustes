from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.io.blob import save_json_gz
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.adapters import PaRoutesAdapter
from retrocast.v2.cli.main import main
from retrocast.v2.io import (
    load_analysis_report,
    load_candidates,
    load_collected_candidates,
    load_evaluation,
    save_benchmark,
    save_candidates,
)
from retrocast.v2.models import Benchmark, Target, TaskConstraints
from retrocast.v2.models.candidates import Candidate
from retrocast.v2.workflow import adapt_candidates


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


def test_v2_config_cli_reports_resolved_data_dir(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"

    run_cli(monkeypatch, "--data-dir", str(data_dir), "config")

    output = capsys.readouterr().out
    assert "RetroCast Schema V2 Configuration" in output
    assert str(data_dir.resolve()) in output.replace("\n", "")
    assert "benchmarks" in output
    assert "processed" in output


def test_v2_config_cli_reports_malformed_yaml_as_cli_error(tmp_path, monkeypatch, caplog) -> None:
    config_path = tmp_path / "retrocast-config.yaml"
    config_path.write_text("data_dir: [unterminated", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        run_cli(monkeypatch, "--config", str(config_path), "config")

    assert exc_info.value.code == 1
    assert "Failed to parse config file" in caplog.text


def test_v2_list_adapters_cli_reports_canonical_names_and_aliases(monkeypatch, capsys) -> None:
    run_cli(monkeypatch, "list-adapters")

    output = capsys.readouterr().out
    assert "paroutes" in output
    assert "retro-star -> retrostar" in output


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


def test_v2_collect_cli_writes_collected_candidates_and_manifest(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "candidates.json.gz"
    benchmark_path = tmp_path / "small.json.gz"
    output_path = tmp_path / "collected.json.gz"
    candidates = load_candidates_from_raw(raw_route())
    save_candidates(candidates, candidates_path)
    save_benchmark(benchmark(), benchmark_path)

    run_cli(
        monkeypatch,
        "collect",
        "--input",
        str(candidates_path),
        "--benchmark",
        str(benchmark_path),
        "--output",
        str(output_path),
    )

    collected = load_collected_candidates(output_path)
    assert list(collected) == ["ethanol"]
    assert len(collected["ethanol"]) == 1
    manifest = json.loads((tmp_path / "collected.manifest.json").read_text(encoding="utf-8"))
    assert manifest["action"] == "[cli:v2]collect"


def test_v2_project_cli_ingest_score_analyze(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "1-benchmarks" / "definitions").mkdir(parents=True)
    write_stock(data_dir / "1-benchmarks" / "stocks" / "test-stock.csv.gz")

    save_benchmark(benchmark(), data_dir / "1-benchmarks" / "definitions" / "small.json.gz")
    write_raw_job(data_dir, model="test-model", dataset="small")

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


def test_v2_ingest_cli_discovers_all_models_and_datasets(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "1-benchmarks" / "definitions").mkdir(parents=True)
    save_benchmark(benchmark(), data_dir / "1-benchmarks" / "definitions" / "small.json.gz")
    write_raw_job(data_dir, model="model-a", dataset="small")
    write_raw_job(data_dir, model="model-b", dataset="small")

    run_cli(monkeypatch, "--data-dir", str(data_dir), "ingest", "--all-models", "--all-datasets", "--no-progress")

    assert (data_dir / "3-processed" / "small" / "model-a" / "candidates.json.gz").exists()
    assert (data_dir / "3-processed" / "small" / "model-b" / "candidates.json.gz").exists()


def load_candidates_from_raw(raw: dict) -> list[Candidate]:
    return adapt_candidates({"ethanol": raw}, PaRoutesAdapter(), target=benchmark().targets["ethanol"])


def write_raw_job(data_dir: Path, *, model: str, dataset: str) -> None:
    raw_dir = data_dir / "2-raw" / model / dataset
    raw_dir.mkdir(parents=True)
    save_json_gz({"ethanol": raw_route()}, raw_dir / "results.json.gz")
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "2", "directives": {"adapter": "paroutes", "raw_results_filename": "results.json.gz"}}
        ),
        encoding="utf-8",
    )
