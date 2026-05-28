from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import ArtifactNotFoundError
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.io import (
    load_benchmark,
    load_collected_candidates,
    load_collected_routes,
    load_task,
    save_benchmark,
    save_collected_candidates,
    save_collected_routes,
    save_task,
)
from retrocast.v2.models import Benchmark, Candidate, FailureRecord, Molecule, Route, Target, Task, TaskConstraints


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
