from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

from retrocast.adapters import PaRoutesAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.cli.main import main
from retrocast.io import (
    load_analysis_report,
    load_candidates,
    load_collected_candidates,
    load_evaluation,
    save_analysis_report,
    save_benchmark,
    save_candidates,
    save_evaluation,
    save_json_gz,
)
from retrocast.models import Benchmark, StockTerminationConstraint, Target
from retrocast.models.analysis import AnalysisReport, MetricSummary
from retrocast.models.candidates import Candidate
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.workflow import adapt_candidates


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
        default_constraints=[StockTerminationConstraint(stock="test-stock")],
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


def test_v2_list_cli_reports_raw_manifests(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    write_raw_job(data_dir, model="test-model", dataset="small")

    run_cli(monkeypatch, "--data-dir", str(data_dir), "list")

    output = capsys.readouterr().out
    assert "test-model" in output
    assert "small" in output
    assert "paroutes" in output


def test_v2_compare_pareto_frontier_uses_analysis_reports(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    for model_name, value in [("model-a", 0.75), ("model-b", 0.5)]:
        analysis_path = data_dir / "5-results" / "small" / model_name / "test-stock" / "analysis.json.gz"
        save_analysis_report(
            AnalysisReport(
                metrics={
                    "solv_0[test-stock]_rate": MetricSummary(
                        value=value,
                        count=4,
                        ci_low=max(0.0, value - 0.1),
                        ci_high=min(1.0, value + 0.1),
                    )
                }
            ),
            analysis_path,
        )

    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "\n".join(
            [
                "benchmark: small",
                "stock: test-stock",
                "metric: solv_0[test-stock]_rate",
                "output_dir: comparisons",
                "output_file: solv0",
                "sources:",
                f"  - root: {data_dir}",
                "    models:",
                "      - name: model-a",
                "        hourly_cost: 1.0",
                "        legend: Model A",
                "      - name: model-b",
                "        hourly_cost: 0.5",
                "        legend: Model B",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_cli(monkeypatch, "compare", "pareto-frontier", str(config_path), "--no-open")

    html_path = tmp_path / "comparisons" / "solv0-cost.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Model A" in html
    assert "solv_0[test-stock]_rate" in html


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
    assert manifest["action"] == "[cli]collect"


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

    score_file_path = tmp_path / "score-file-evaluation.json.gz"
    run_cli(
        monkeypatch,
        "score-file",
        "--benchmark",
        str(data_dir / "1-benchmarks" / "definitions" / "small.json.gz"),
        "--candidates",
        str(candidates_path),
        "--stock",
        str(data_dir / "1-benchmarks" / "stocks" / "test-stock.csv.gz"),
        "--output",
        str(score_file_path),
        "--model-name",
        "test-model",
        "--acceptable-route-match",
        "exact",
    )
    score_file_evaluation = load_evaluation(score_file_path)
    assert score_file_evaluation.acceptable_route_match == "exact"
    assert score_file_evaluation.targets["ethanol"].candidates[0].satisfies_task()
    score_file_manifest = json.loads((tmp_path / "score-file-evaluation.manifest.json").read_text(encoding="utf-8"))
    assert score_file_manifest["parameters"]["acceptable_route_match"] == "exact"

    run_cli(monkeypatch, "--data-dir", str(data_dir), "score", "--model", "test-model", "--dataset", "small")
    evaluation_path = data_dir / "4-scored" / "small" / "test-model" / "test-stock" / "evaluation.json.gz"
    evaluation = load_evaluation(evaluation_path)
    assert evaluation.schema_version == "2"
    assert evaluation.acceptable_route_match == "prefix"
    assert evaluation.targets["ethanol"].candidates[0].satisfies_task()
    assert evaluation.targets["ethanol"].wall_time == 12.5
    assert evaluation.targets["ethanol"].cpu_time == 3.25

    stale_targets = {"ethanol": evaluation.targets["ethanol"].model_copy(update={"wall_time": None, "cpu_time": None})}
    save_evaluation(evaluation.model_copy(update={"targets": stale_targets}), evaluation_path)

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
    assert report.runtime.total_wall_time == 12.5
    assert report.runtime.total_cpu_time == 3.25
    assert (analysis_path.parent / "report.md").exists()
    manifest_path = analysis_path.parent / "manifest.json"
    assert manifest_path.exists()

    run_cli(monkeypatch, "--data-dir", str(data_dir), "verify", "--target", str(manifest_path))


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
    save_json_gz({"wall_time": {"ethanol": 12.5}, "cpu_time": {"ethanol": 3.25}}, raw_dir / "execution_stats.json.gz")
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "2", "directives": {"adapter": "paroutes", "raw_results_filename": "results.json.gz"}}
        ),
        encoding="utf-8",
    )
