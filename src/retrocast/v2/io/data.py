from __future__ import annotations

import gzip
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from retrocast.exceptions import ArtifactDecodeError, ArtifactFormatError, ArtifactNotFoundError, ArtifactWriteError
from retrocast.v2.models.task import Benchmark, Task
from retrocast.v2.workflow.collect import CollectedCandidates, CollectedRoutes

_TASK_ADAPTER = TypeAdapter(Task)
_BENCHMARK_ADAPTER = TypeAdapter(Benchmark)
_COLLECTED_ROUTES_ADAPTER = TypeAdapter(CollectedRoutes)
_COLLECTED_CANDIDATES_ADAPTER = TypeAdapter(CollectedCandidates)

T = TypeVar("T")


def load_task(path: Path) -> Task:
    return _load_model(path, _TASK_ADAPTER, artifact="task")


def save_task(task: Task, path: Path) -> None:
    _save_model(task, path, _TASK_ADAPTER, artifact="task")


def load_benchmark(path: Path) -> Benchmark:
    return _load_model(path, _BENCHMARK_ADAPTER, artifact="benchmark")


def save_benchmark(benchmark: Benchmark, path: Path) -> None:
    _save_model(benchmark, path, _BENCHMARK_ADAPTER, artifact="benchmark")


def load_collected_routes(path: Path) -> CollectedRoutes:
    return _load_model(path, _COLLECTED_ROUTES_ADAPTER, artifact="collected_routes")


def save_collected_routes(routes: CollectedRoutes, path: Path) -> None:
    _save_model(routes, path, _COLLECTED_ROUTES_ADAPTER, artifact="collected_routes")


def load_collected_candidates(path: Path) -> CollectedCandidates:
    return _load_model(path, _COLLECTED_CANDIDATES_ADAPTER, artifact="collected_candidates")


def save_collected_candidates(candidates: CollectedCandidates, path: Path) -> None:
    _save_model(candidates, path, _COLLECTED_CANDIDATES_ADAPTER, artifact="collected_candidates")


def _load_model(path: Path, adapter: TypeAdapter[T], *, artifact: str) -> T:
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"{artifact} file not found: {path}",
            code="io.not_found",
            context={"path": str(path), "artifact": artifact},
        )

    try:
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    except (OSError, EOFError) as exc:
        raise ArtifactDecodeError(
            f"Failed to load {artifact} from {path}: {exc}",
            code="io.decode_failed",
            context={"path": str(path), "artifact": artifact},
        ) from exc

    try:
        return adapter.validate_json(payload)
    except ValidationError as exc:
        raise ArtifactFormatError(
            f"Invalid {artifact} JSON format in {path}: {exc}",
            code="io.invalid_artifact_shape",
            context={"path": str(path), "artifact": artifact},
        ) from exc


def _save_model(value: T, path: Path, adapter: TypeAdapter[T], *, artifact: str) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = adapter.dump_json(value, indent=2, exclude_none=True, exclude_computed_fields=True)
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactWriteError(
            f"Failed to save {artifact} to {path}: {exc}",
            code="io.write_failed",
            context={"path": str(path), "artifact": artifact},
        ) from exc
