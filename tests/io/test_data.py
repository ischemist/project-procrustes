from __future__ import annotations

import gzip
from typing import Literal

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import ArtifactDecodeError, ArtifactFormatError, ArtifactNotFoundError, ArtifactWriteError
from retrocast.io import (
    load_benchmark,
    load_candidates,
    load_collected_candidates,
    load_collected_routes,
    load_evaluation,
    load_execution_stats,
    load_json_artifact,
    load_routes,
    load_stock_file,
    load_task,
    save_analysis_report,
    save_benchmark,
    save_candidates,
    save_collected_candidates,
    save_collected_routes,
    save_csv_gz,
    save_evaluation,
    save_execution_stats,
    save_json_gz,
    save_jsonl_gz,
    save_lines_gz,
    save_routes,
    save_stock_files,
    save_task,
)
from retrocast.io.data import BenchmarkResultsLoader
from retrocast.models import (
    Benchmark,
    Candidate,
    FailureRecord,
    Molecule,
    Route,
    RouteDepthConstraint,
    StockTerminationConstraint,
    Target,
    Task,
)
from retrocast.models.analysis import AnalysisReport, MetricSummary
from retrocast.models.evaluation import (
    CheckStatus,
    ConstraintResult,
    Evaluation,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.utils.timing import ExecutionStats


def target(smiles: str = "CCO", target_id: str = "ethanol") -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)))


def route(smiles: str = "CCO") -> Route:
    canonical = canonicalize_smiles(smiles)
    return Route(target=Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical))))


def benchmark() -> Benchmark:
    benchmark_target = target()
    return Benchmark(
        name="small",
        description="round-trip fixture",
        targets={benchmark_target.id: benchmark_target},
        default_constraints=[StockTerminationConstraint(stock="n5-stock")],
    )


def task_value() -> Task:
    task_target = target()
    return Task(name="small-task", targets={task_target.id: task_target})


def test_task_round_trips(tmp_path) -> None:
    path = tmp_path / "task.json.gz"
    value = task_value()

    save_task(value, path)

    assert load_task(path) == value


def test_schema_model_writes_are_deterministic(tmp_path) -> None:
    left = tmp_path / "left.json.gz"
    right = tmp_path / "right.json.gz"
    value = task_value()

    save_task(value, left)
    save_task(value, right)

    assert left.read_bytes() == right.read_bytes()


def test_stock_file_writes_are_deterministic(tmp_path) -> None:
    stock = {
        InChIKeyStr(get_inchi_key("C")): SmilesStr("C"),
        InChIKeyStr(get_inchi_key("CO")): SmilesStr("CO"),
    }

    left_csv, left_txt, _ = save_stock_files(stock, "tiny", tmp_path / "left" / "stocks")
    right_csv, right_txt, _ = save_stock_files(stock, "tiny", tmp_path / "right" / "stocks")

    assert left_csv.read_bytes() == right_csv.read_bytes()
    assert left_txt.read_bytes() == right_txt.read_bytes()


def test_load_json_artifact_dispatches_supported_formats_and_reports_jsonl_rows(tmp_path) -> None:
    plain_json = tmp_path / "artifact.json"
    gzip_json = tmp_path / "artifact.json.gz"
    plain_jsonl = tmp_path / "artifact.jsonl"
    gzip_jsonl = tmp_path / "artifact.jsonl.gz"
    invalid_jsonl = tmp_path / "invalid.jsonl"
    unsupported = tmp_path / "artifact.txt"

    plain_json.write_text('{"kind": "plain"}', encoding="utf-8")
    save_json_gz({"kind": "gzip"}, gzip_json)
    plain_jsonl.write_text('{"row": 1}\n\n{"row": 2}\n', encoding="utf-8")
    save_jsonl_gz([{"row": 3}, {"row": 4}], gzip_jsonl)
    invalid_jsonl.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    unsupported.write_text("{}", encoding="utf-8")

    assert load_json_artifact(plain_json) == {"kind": "plain"}
    assert load_json_artifact(gzip_json) == {"kind": "gzip"}
    assert load_json_artifact(plain_jsonl) == [{"row": 1}, {"row": 2}]
    assert load_json_artifact(gzip_jsonl) == [{"row": 3}, {"row": 4}]
    with pytest.raises(ArtifactDecodeError) as row_error:
        load_json_artifact(invalid_jsonl)
    assert row_error.value.context["line_number"] == 2
    with pytest.raises(ArtifactDecodeError) as suffix_error:
        load_json_artifact(unsupported)
    assert suffix_error.value.code == "io.decode_failed"


def test_load_stock_file_reads_columns_and_rejects_bad_inputs(tmp_path) -> None:
    stock = {
        InChIKeyStr(get_inchi_key("C")): SmilesStr("C"),
        InChIKeyStr(get_inchi_key("CO")): SmilesStr("CO"),
    }
    csv_path, _, _ = save_stock_files(stock, "tiny", tmp_path / "stocks")
    wrong_suffix = tmp_path / "stock.csv"
    missing_column = tmp_path / "missing-column.csv.gz"
    corrupt = tmp_path / "corrupt.csv.gz"

    wrong_suffix.write_text("SMILES,InChIKey\n", encoding="utf-8")
    with gzip.open(missing_column, "wt", encoding="utf-8") as handle:
        handle.write("SMILES\nC\n")
    corrupt.write_bytes(b"not gzip")

    assert load_stock_file(csv_path) == set(stock)
    assert load_stock_file(csv_path, return_as="smiles") == set(stock.values())
    with pytest.raises(ValueError):
        load_stock_file(csv_path, return_as="bad")
    with pytest.raises(ArtifactNotFoundError):
        load_stock_file(tmp_path / "missing.csv.gz")
    with pytest.raises(ArtifactFormatError):
        load_stock_file(wrong_suffix)
    with pytest.raises(ArtifactFormatError) as column_error:
        load_stock_file(missing_column)
    assert column_error.value.context["required_column"] == "InChIKey"
    with pytest.raises(ArtifactDecodeError):
        load_stock_file(corrupt)


@pytest.mark.parametrize(
    ("writer", "rows", "filename"),
    [
        (save_jsonl_gz, [{"row": 1}], "rows.jsonl.gz"),
        (save_lines_gz, ["line"], "lines.txt.gz"),
        (save_csv_gz, [["cell"]], "rows.csv.gz"),
    ],
)
def test_stream_writers_wrap_parent_directory_failures(tmp_path, writer, rows, filename) -> None:
    parent_file = tmp_path / "not-a-dir"
    parent_file.write_text("occupied", encoding="utf-8")
    path = parent_file / filename

    with pytest.raises(ArtifactWriteError) as exc_info:
        writer(rows, path)

    assert exc_info.value.code == "io.write_failed"
    assert exc_info.value.context["path"] == str(path)


def test_benchmark_round_trips(tmp_path) -> None:
    path = tmp_path / "benchmark.json.gz"
    value = benchmark()

    save_benchmark(value, path)

    assert load_benchmark(path) == value


def test_collected_routes_round_trip(tmp_path) -> None:
    path = tmp_path / "routes.json.gz"
    value = {"ethanol": [route()]}

    save_collected_routes(value, path)

    assert load_collected_routes(path) == value


def test_collected_candidates_round_trip(tmp_path) -> None:
    path = tmp_path / "candidates.json.gz"
    value = {
        "ethanol": [
            Candidate(rank=1, route=route()),
            Candidate(rank=2, failure=FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="ethanol")),
        ]
    }

    save_collected_candidates(value, path)

    assert load_collected_candidates(path) == value


def test_route_candidate_evaluation_analysis_and_execution_artifacts_round_trip(tmp_path) -> None:
    routes_path = tmp_path / "routes.json.gz"
    candidates_path = tmp_path / "candidates.json.gz"
    evaluation_path = tmp_path / "evaluation.json.gz"
    analysis_path = tmp_path / "analysis.json.gz"
    execution_path = tmp_path / "execution.json.gz"

    routes = [route()]
    candidates = [Candidate(rank=1, route=route())]
    evaluation = scored_evaluation()
    analysis = AnalysisReport(metrics={"solv_0": MetricSummary(value=1.0, count=1)})
    execution = ExecutionStats(wall_time={"ethanol": 1.5}, cpu_time={"ethanol": 1.0})

    save_routes(routes, routes_path)
    save_candidates(candidates, candidates_path)
    save_evaluation(evaluation, evaluation_path)
    save_analysis_report(analysis, analysis_path)
    save_execution_stats(execution, execution_path)

    assert load_routes(routes_path) == routes
    assert load_candidates(candidates_path) == candidates
    assert load_evaluation(evaluation_path) == evaluation
    assert load_json_artifact(analysis_path) == analysis.model_dump(mode="json", exclude_none=True)
    assert load_execution_stats(execution_path) == execution


def test_benchmark_results_loader_exposes_evaluation_sequence_and_statistics(tmp_path) -> None:
    data_dir = tmp_path / "retrocast"
    evaluation_path = data_dir / "4-scored" / "small" / "model-a" / "stock-a" / "evaluation.json.gz"
    evaluation = scored_evaluation()
    save_evaluation(evaluation, evaluation_path)

    loader = BenchmarkResultsLoader(tmp_path)
    loaded = loader.load_evaluation("small", "model-a", "stock-a")
    stats = loader.load_statistics("small", ["model-a", "missing"], "stock-a")

    assert loaded is not None
    assert loaded.results == evaluation.targets
    assert len(loaded) == 1
    assert list(loaded) == list(evaluation.targets.values())
    assert loaded[0] == next(iter(evaluation.targets.values()))
    assert loader.load_evaluation("small", "missing", "stock-a") is None
    assert len(stats) == 1
    assert stats[0].model_name == "model-a"
    assert stats[0].solv_0 is stats[0].stock_termination
    assert stats[0].top_k_accuracy[1].overall.value == 1.0
    assert stats[0].top_k_accuracy[1].by_stratum["depth 1"].count == 1


def test_benchmark_results_loader_keeps_named_route_depth_strata(tmp_path) -> None:
    data_dir = tmp_path / "retrocast"
    evaluation_path = data_dir / "4-scored" / "small" / "model-a" / "stock-a" / "evaluation.json.gz"
    save_evaluation(scored_evaluation(route_depth="short"), evaluation_path)

    stats = BenchmarkResultsLoader(tmp_path).load_statistics("small", ["model-a"], "stock-a")

    assert stats[0].top_k_accuracy[1].by_stratum["depth short"].count == 1


def test_missing_benchmark_raises_io_error(tmp_path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        load_benchmark(tmp_path / "missing.json.gz")


def test_invalid_gzip_raises_decode_error(tmp_path) -> None:
    path = tmp_path / "benchmark.json.gz"
    path.write_text("not gzip", encoding="utf-8")

    with pytest.raises(ArtifactDecodeError):
        load_benchmark(path)


def test_truncated_gzip_raises_decode_error(tmp_path) -> None:
    path = tmp_path / "benchmark.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"{}")
    path.write_bytes(path.read_bytes()[:-4])

    with pytest.raises(ArtifactDecodeError):
        load_benchmark(path)


def test_invalid_benchmark_shape_raises_format_error(tmp_path) -> None:
    path = tmp_path / "benchmark.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"{}")

    with pytest.raises(ArtifactFormatError):
        load_benchmark(path)


def test_save_failure_raises_write_error(tmp_path) -> None:
    parent_file = tmp_path / "not-a-dir"
    parent_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(ArtifactWriteError):
        save_benchmark(benchmark(), parent_file / "benchmark.json.gz")


def scored_evaluation(route_depth: int | Literal["short", "medium", "long"] = 1) -> Evaluation:
    benchmark_target = target()
    task = Task(name="small-task", targets={benchmark_target.id: benchmark_target})
    scored = ScoredCandidate(
        rank=1,
        route=route(),
        validity=RouteValidity(tiers={Tier.ZERO: TierResult(status=CheckStatus.PASS)}),
        constraints=ConstraintResult(status=CheckStatus.PASS),
        matches_acceptable=True,
    )
    target_result = TargetResult(
        target=benchmark_target,
        effective_constraints=[RouteDepthConstraint(max_depth=route_depth)],
        candidates=[scored],
        wall_time=1.0,
        cpu_time=0.5,
    )
    return Evaluation(task=task, tiers=[Tier.ZERO], targets={benchmark_target.id: target_result})
