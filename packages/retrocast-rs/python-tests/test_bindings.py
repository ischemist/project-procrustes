from __future__ import annotations

import importlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import retrocast

FIXTURE = Path(__file__).parents[1] / "test-data" / "aizynth-smoke.json"
TARGET_ID = "ethanol"


def task() -> dict[str, object]:
    smiles = retrocast.canonicalize_smiles("CCO")
    return {
        "name": "python-binding-contract",
        "targets": {
            TARGET_ID: {
                "id": TARGET_ID,
                "smiles": smiles,
                "inchikey": retrocast.get_inchi_key(smiles),
            }
        },
        "default_constraints": [{"kind": "retrocast.stock_termination", "stock": "test-stock"}],
    }


def stock() -> dict[str, list[str]]:
    return {
        "test-stock": [
            retrocast.get_inchi_key("C"),
            retrocast.get_inchi_key("O"),
        ]
    }


def raw() -> list[object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_distribution_is_the_native_module() -> None:
    assert retrocast.__engine__ == "rust"
    assert isinstance(retrocast.__version__, str)
    assert retrocast.__version__
    assert retrocast.engine_info()[0] == retrocast.__version__

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("retrocast.adapters")

    assert set(retrocast.__all__) == {
        "NativeEvaluation",
        "NativePredictions",
        "__engine__",
        "__version__",
        "adapt",
        "analyze",
        "analyze_file",
        "canonicalize_smiles",
        "create_manifest",
        "create_planner_manifest",
        "engine_info",
        "evaluate",
        "get_inchi_key",
        "ingest",
        "ingest_file",
        "load_stock",
        "load_task",
        "molecular_descriptors",
        "read_json",
        "reduce_inchi_key",
        "resolve_stock_bindings",
        "score",
        "validate_execution_stats",
        "validate_task",
        "verify_manifest",
        "verify_planner_manifest",
        "write_execution_stats",
        "write_json",
        "write_json_gz",
        "write_task",
    }


def test_chemistry_uses_bundled_rdkit_cpp() -> None:
    assert retrocast.engine_info()[1] == "RDKit C++"
    assert retrocast.canonicalize_smiles("OCC") == "CCO"
    assert retrocast.get_inchi_key("CCO") == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    with pytest.raises(ValueError, match="SMILES"):
        retrocast.canonicalize_smiles("not a molecule")


def test_adapt_accepts_plain_python_values() -> None:
    candidates = retrocast.adapt(raw(), "aizynthfinder", workers=2)

    assert len(candidates) == 1
    assert candidates[0]["rank"] == 1
    assert candidates[0]["route"]["target"]["smiles"] == "CCO"


@pytest.mark.parametrize("adapter", ["synplanner", "syntheseus"])
@pytest.mark.parametrize(
    ("mode", "payload", "max_candidates", "expected_failure_codes"),
    [
        pytest.param(
            "strict",
            [{"type": "mol", "smiles": "CCO"}, {"type": "mol", "smiles": "OCC"}],
            None,
            [None, None],
            id="all-valid-strict",
        ),
        pytest.param(
            "prune",
            [{"type": "mol", "smiles": "CCO"}, {"type": "mol", "smiles": "OCC"}],
            None,
            [None, None],
            id="all-valid-prune",
        ),
        pytest.param(
            "strict",
            [
                {"type": "mol", "smiles": "CCO"},
                {"type": "mol", "smiles": "CCO", "children": [None]},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            None,
            [None, "adapter.schema_invalid", "chem.invalid_smiles"],
            id="mixed-strict",
        ),
        pytest.param(
            "prune",
            [
                {"type": "mol", "smiles": "CCO"},
                {"type": "mol", "smiles": "CCO", "children": [None]},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            None,
            [None, "adapter.schema_invalid", "adapter.target_pruned"],
            id="mixed-prune",
        ),
        pytest.param(
            "strict",
            [
                {"type": "mol", "smiles": "CCO", "children": [True]},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            None,
            ["adapter.schema_invalid", "chem.invalid_smiles"],
            id="all-invalid-strict",
        ),
        pytest.param(
            "prune",
            [
                {"type": "mol", "smiles": "CCO", "children": [True]},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            None,
            ["adapter.schema_invalid", "adapter.target_pruned"],
            id="all-invalid-prune",
        ),
        pytest.param(
            "strict",
            [
                {"type": "mol", "smiles": "CCO", "children": [None]},
                {"type": "mol", "smiles": "OCC"},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            2,
            ["adapter.schema_invalid", None],
            id="max-candidates-strict",
        ),
        pytest.param(
            "prune",
            [
                {"type": "mol", "smiles": "CCO", "children": [None]},
                {"type": "mol", "smiles": "OCC"},
                {"type": "mol", "smiles": "not-smiles"},
            ],
            2,
            ["adapter.schema_invalid", None],
            id="max-candidates-prune",
        ),
    ],
)
def test_bipartite_candidate_slot_matrix(
    adapter: str,
    mode: str,
    payload: list[dict[str, object]],
    max_candidates: int | None,
    expected_failure_codes: list[str | None],
) -> None:
    target = task()["targets"][TARGET_ID]
    candidates = retrocast.adapt(
        payload,
        adapter,
        mode=mode,
        target=target,
        source_key="source-ethanol",
        max_candidates=max_candidates,
    )

    assert [candidate["rank"] for candidate in candidates] == list(range(1, len(expected_failure_codes) + 1))
    for candidate, expected_failure_code in zip(candidates, expected_failure_codes, strict=True):
        if expected_failure_code is None:
            assert candidate["route"]["target"]["smiles"] == "CCO"
            assert "failure" not in candidate
        else:
            assert candidate["failure"]["code"] == expected_failure_code
            assert candidate["failure"]["target_id"] == TARGET_ID
            assert "route" not in candidate


def test_native_handles_keep_composed_state_in_rust(tmp_path: Path) -> None:
    predictions = retrocast.ingest(
        {TARGET_ID: raw()},
        "aizynthfinder",
        task(),
        workers=2,
    )
    assert isinstance(predictions, retrocast.NativePredictions)
    assert predictions.to_dict()[TARGET_ID][0]["rank"] == 1

    predictions_path = tmp_path / "candidates.json.gz"
    predictions.write(predictions_path)
    assert predictions_path.is_file()

    evaluation = retrocast.score(predictions, task(), stock(), workers=2)
    assert isinstance(evaluation, retrocast.NativeEvaluation)
    with pytest.raises(RuntimeError, match="consumed by score"):
        predictions.to_dict()
    assert evaluation.metric_label() == "test-stock"
    candidate = evaluation.to_dict()["targets"][TARGET_ID]["candidates"][0]
    assert candidate["constraints"]["status"] == "pass"

    evaluation_path = tmp_path / "evaluation.json.gz"
    evaluation.write(evaluation_path)
    assert evaluation_path.is_file()

    report = retrocast.analyze(evaluation, n_boot=16, workers=2)
    assert report["metrics"]["solv_0[test-stock]_rate"]["value"] == 1.0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires a POSIX FIFO")
def test_failed_score_does_not_consume_predictions_while_write_is_active(tmp_path: Path) -> None:
    predictions = retrocast.ingest(
        {TARGET_ID: raw() * 2_048},
        "aizynthfinder",
        task(),
        workers=2,
    )
    fifo = tmp_path / "predictions.pipe"
    os.mkfifo(fifo)
    reader_opened = threading.Event()
    release_reader = threading.Event()

    def drain_fifo() -> None:
        with fifo.open("rb") as stream:
            reader_opened.set()
            release_reader.wait(timeout=5)
            while stream.read(64 * 1_024):
                pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(predictions.write, fifo)
        reader = pool.submit(drain_fifo)
        try:
            assert reader_opened.wait(timeout=5)
            with pytest.raises(RuntimeError, match="currently in use"):
                retrocast.score(predictions, task(), stock(), workers=2)
            assert predictions.to_dict()[TARGET_ID][0]["rank"] == 1
        finally:
            release_reader.set()
        writer.result(timeout=5)
        reader.result(timeout=5)

    evaluation = retrocast.score(predictions, task(), stock(), workers=2)
    assert isinstance(evaluation, retrocast.NativeEvaluation)


def test_evaluate_files_runs_without_python_materialization(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    stock_path = tmp_path / "test-stock.txt"
    output_dir = tmp_path / "output"
    benchmark_path.write_text(json.dumps(task()), encoding="utf-8")
    stock_path.write_text("C\nO\n", encoding="utf-8")

    stats = retrocast.evaluate(
        FIXTURE,
        benchmark_path,
        stock_path,
        output_dir,
        n_boot=16,
        workers=2,
    )

    assert stats["engine"] == "rust"
    assert stats["targets"] == 1
    assert stats["candidates"] == 1
    assert (output_dir / "candidates.json.gz").is_file()
    assert (output_dir / "evaluation.json.gz").is_file()
    assert (output_dir / "analysis.json.gz").is_file()
    assert (output_dir / "evaluation-run.json").is_file()


@pytest.mark.parametrize("entrypoint", ["adapt", "ingest"])
def test_worker_count_must_be_positive(entrypoint: str) -> None:
    with pytest.raises(RuntimeError, match="worker"):
        if entrypoint == "adapt":
            retrocast.adapt(raw(), "aizynthfinder", workers=0)
        else:
            retrocast.ingest(raw(), "aizynthfinder", task(), workers=0)


def test_unknown_adapter_fails_at_the_binding_boundary() -> None:
    with pytest.raises(RuntimeError, match="unknown RetroCast adapter"):
        retrocast.adapt(raw(), "missing-adapter")


def test_task_loading_and_validation_return_normalized_dicts(tmp_path: Path) -> None:
    path = tmp_path / "task.json.gz"
    retrocast.write_json_gz(task(), path)

    loaded = retrocast.load_task(path)
    validated = retrocast.validate_task(task())

    assert loaded == validated
    assert loaded["schema_version"] == "2"
    assert loaded["targets"][TARGET_ID]["id"] == TARGET_ID

    invalid = task()
    invalid["targets"] = {"wrong-key": invalid["targets"][TARGET_ID]}
    with pytest.raises(ValueError, match="does not match"):
        retrocast.validate_task(invalid)

    mismatched = task()
    mismatched["targets"][TARGET_ID]["inchikey"] = retrocast.get_inchi_key("C")
    retrocast.write_json_gz(mismatched, path)
    assert retrocast.load_task(path)["targets"][TARGET_ID]["smiles"] == "CCO"
    assert retrocast.validate_task(mismatched, chemistry=False)["schema_version"] == "2"
    with pytest.raises(ValueError, match="does not match"):
        retrocast.load_task(path, chemistry=True)
    with pytest.raises(ValueError, match="does not match"):
        retrocast.validate_task(mismatched)
    with pytest.raises(ValueError, match="does not match"):
        retrocast.write_task(mismatched, tmp_path / "invalid-task.json.gz")

    retrocast.write_task(task(), path)
    assert retrocast.load_task(path) == validated

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        retrocast.load_task(malformed_path)


def test_stock_bindings_apply_task_override_rules() -> None:
    value = task()
    value["targets"]["methane"] = {
        "id": "methane",
        "smiles": "C",
        "inchikey": retrocast.get_inchi_key("C"),
    }
    value["constraints"] = {"methane": [{"kind": "retrocast.stock_termination", "stock": "special-stock"}]}

    assert retrocast.resolve_stock_bindings(value) == {
        TARGET_ID: "test-stock",
        "methane": "special-stock",
    }


def test_producer_artifacts_use_rust_io_and_provenance(tmp_path: Path) -> None:
    source_path = tmp_path / "task.json.gz"
    results_path = tmp_path / "results.json.gz"
    manifest_path = tmp_path / "manifest.json"
    results = {TARGET_ID: raw()}

    retrocast.write_json_gz(task(), source_path)
    retrocast.write_json_gz(results, results_path)
    first_bytes = results_path.read_bytes()
    retrocast.write_json_gz(results, results_path)
    assert results_path.read_bytes() == first_bytes
    assert retrocast.read_json(results_path) == results

    manifest = retrocast.create_manifest(
        "planner-run",
        [source_path],
        [
            {
                "path": results_path,
                "content_type": "unknown",
            }
        ],
        tmp_path,
        directives={"adapter": "aizynthfinder"},
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_manifest(manifest_path, tmp_path)

    assert manifest["action"] == "planner-run"
    assert manifest["directives"]["adapter"] == "aizynthfinder"
    assert report["is_valid"] is True


def test_manifest_creation_requires_existing_sources_and_outputs(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="manifest source file not found"):
        retrocast.create_manifest(
            "planner-run",
            [tmp_path / "missing.json"],
            [],
            tmp_path,
        )

    with pytest.raises(OSError, match="manifest output file not found"):
        retrocast.create_manifest(
            "planner-run",
            [],
            [{"path": tmp_path / "missing.json", "content_type": "unknown"}],
            tmp_path,
        )

    output = tmp_path / "output.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="value is required"):
        retrocast.create_manifest(
            "known-content",
            [],
            [{"path": output, "content_type": "benchmark"}],
            tmp_path,
        )

    manifest = retrocast.create_manifest(
        "prehashed-content",
        [],
        [
            {
                "path": output,
                "content_type": "benchmark",
                "content_hash": "sha256:producer-supplied",
            }
        ],
        tmp_path,
    )
    assert manifest["output_files"][0]["content_hash"] == "sha256:producer-supplied"


def test_manifest_verification_is_strict_by_default(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    manifest_path = tmp_path / "manifest.json"
    output.write_text("{}", encoding="utf-8")
    manifest = retrocast.create_manifest(
        "producer",
        [],
        [{"path": output, "content_type": "unknown"}],
        tmp_path,
    )
    retrocast.write_json(manifest, manifest_path)
    output.unlink()

    assert retrocast.verify_manifest(manifest_path, tmp_path)["is_valid"] is False
    assert retrocast.verify_manifest(manifest_path, tmp_path, lenient=True)["is_valid"] is True


def test_planner_manifest_is_project_ingest_compatible(tmp_path: Path) -> None:
    raw_dir = tmp_path / "2-raw" / "model" / "task"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "routes.json.gz"
    source_path = tmp_path / "task.json"
    manifest_path = raw_dir / "manifest.json"
    raw_path.write_bytes(b"raw")
    source_path.write_text("{}", encoding="utf-8")

    manifest = retrocast.create_planner_manifest(
        "planner-run",
        "retro-star",
        raw_path,
        [source_path],
        tmp_path,
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, tmp_path)

    assert manifest["directives"] == {
        "adapter": "retrostar",
        "raw_results_filename": "routes.json.gz",
    }
    assert report["is_valid"] is True

    manifest["directives"].pop("adapter")
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, tmp_path)
    assert report["is_valid"] is False
    assert any("not ingestible" in issue["message"] for issue in report["issues"])


def test_stock_loader_selects_the_planner_representation(tmp_path: Path) -> None:
    stock_path = tmp_path / "stock.csv"
    stock_path.write_text(
        "SMILES,InChIKey\nCCO,LFQSCWFLJHTTHZ-UHFFFAOYSA-N\nC,VNWKTOKETHGBQD-UHFFFAOYSA-N\n",
        encoding="utf-8",
    )

    assert retrocast.load_stock(stock_path) == ["C", "CCO"]
    assert retrocast.load_stock(stock_path, representation="inchikey") == [
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "VNWKTOKETHGBQD-UHFFFAOYSA-N",
    ]

    missing_column = tmp_path / "missing-column.csv"
    missing_column.write_text("molecule\nCCO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stock CSV has no smiles column"):
        retrocast.load_stock(missing_column)

    invalid_smiles = tmp_path / "invalid.smi"
    invalid_smiles.write_text("not-smiles\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid SMILES"):
        retrocast.load_stock(invalid_smiles, representation="inchikey")


def test_execution_stats_are_validated_before_writing(tmp_path: Path) -> None:
    stats = {"wall_time": {TARGET_ID: 1.5}, "cpu_time": {TARGET_ID: 1.2}}
    gzip_path = tmp_path / "execution_stats.json.gz"
    json_path = tmp_path / "execution_stats.json"

    assert retrocast.validate_execution_stats(stats) == stats
    retrocast.write_execution_stats(stats, gzip_path)
    retrocast.write_execution_stats(stats, json_path)
    assert retrocast.read_json(gzip_path) == stats
    assert retrocast.read_json(json_path) == stats
    assert json_path.read_bytes().startswith(b"{")

    with pytest.raises(ValueError, match="non-negative"):
        retrocast.validate_execution_stats({"wall_time": {TARGET_ID: -1}})
