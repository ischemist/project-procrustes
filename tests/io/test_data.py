from __future__ import annotations

import gzip

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import ArtifactDecodeError, ArtifactFormatError, ArtifactNotFoundError, ArtifactWriteError
from retrocast.io import (
    load_benchmark,
    load_collected_candidates,
    load_collected_routes,
    load_task,
    save_benchmark,
    save_collected_candidates,
    save_collected_routes,
    save_stock_files,
    save_task,
)
from retrocast.models import Benchmark, Candidate, FailureRecord, Molecule, Route, Target, Task, TaskConstraints
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr


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
        default_constraints=TaskConstraints(stock="n5-stock"),
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
