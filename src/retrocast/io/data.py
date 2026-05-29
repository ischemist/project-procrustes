from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, overload

from pydantic import TypeAdapter, ValidationError

from retrocast.exceptions import ArtifactDecodeError, ArtifactFormatError, ArtifactNotFoundError, ArtifactWriteError
from retrocast.io.blob import (
    iter_jsonl_gz,
    load_json_gz,
    save_csv_gz,
    save_json_gz,
    save_lines_gz,
)
from retrocast.io.provenance import create_manifest
from retrocast.models.analysis import AnalysisReport
from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import Evaluation
from retrocast.models.route import Route
from retrocast.models.task import Benchmark, Task
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.workflow.collect import CollectedCandidates, CollectedRoutes

_TASK_ADAPTER = TypeAdapter(Task)
_BENCHMARK_ADAPTER = TypeAdapter(Benchmark)
_ROUTE_LIST_ADAPTER = TypeAdapter(list[Route])
_CANDIDATE_LIST_ADAPTER = TypeAdapter(list[Candidate])
_COLLECTED_ROUTES_ADAPTER = TypeAdapter(CollectedRoutes)
_COLLECTED_CANDIDATES_ADAPTER = TypeAdapter(CollectedCandidates)
_EVALUATION_ADAPTER = TypeAdapter(Evaluation)
_ANALYSIS_REPORT_ADAPTER = TypeAdapter(AnalysisReport)
_RAW_PAROUTES_LIST_ADAPTER = TypeAdapter(list[dict])

T = TypeVar("T")


class ManifestStatistics(Protocol):
    def to_manifest_dict(self) -> dict[str, Any]: ...


def load_raw_paroutes_list(path: Path) -> list[dict]:
    return _RAW_PAROUTES_LIST_ADAPTER.validate_python(load_json_gz(path))


def load_training_route_records(path: Path):
    from retrocast.curation.training.records import TrainingRouteRecord

    return [TrainingRouteRecord.model_validate(row) for row in iter_jsonl_gz(path)]


def save_stock_files(
    stock: dict[InChIKeyStr, SmilesStr],
    stock_name: str,
    output_dir: Path,
    source_path: Path | None = None,
    statistics: ManifestStatistics | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stock_name}.csv.gz"
    txt_path = output_dir / f"{stock_name}.txt.gz"
    manifest_path = output_dir / f"{stock_name}.manifest.json"
    sorted_items = sorted(stock.items(), key=lambda item: item[0])

    try:
        save_csv_gz(_stock_csv_rows(sorted_items), csv_path)
        save_lines_gz((str(smiles) for _, smiles in sorted_items), txt_path)
    except ArtifactWriteError as exc:
        raise ArtifactWriteError(
            f"failed to write stock files for {stock_name}: {exc}",
            code=exc.code,
            context={"stock_name": stock_name, "output_dir": str(output_dir), "path": exc.context.get("path")},
        ) from exc

    manifest = create_manifest(
        action="canonicalize-stock",
        sources=[source_path] if source_path else [],
        outputs=[("csv", csv_path, stock, "stock"), ("txt", txt_path, stock, "stock")],
        root_dir=output_dir.parent.parent,
        parameters={"stock_name": stock_name},
        statistics=statistics.to_manifest_dict() if statistics is not None else {},
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return csv_path, txt_path, manifest_path


def _stock_csv_rows(items: list[tuple[InChIKeyStr, SmilesStr]]) -> Iterator[list[str]]:
    yield ["SMILES", "InChIKey"]
    for inchikey, smiles in items:
        yield [str(smiles), str(inchikey)]


def load_task(path: Path) -> Task:
    return _load_model(path, _TASK_ADAPTER, artifact="task")


def save_task(task: Task, path: Path) -> None:
    _save_model(task, path, _TASK_ADAPTER, artifact="task")


def load_benchmark(path: Path) -> Benchmark:
    return _load_model(path, _BENCHMARK_ADAPTER, artifact="benchmark")


def save_benchmark(benchmark: Benchmark, path: Path) -> None:
    _save_model(benchmark, path, _BENCHMARK_ADAPTER, artifact="benchmark")


def load_routes(path: Path) -> list[Route]:
    return _load_model(path, _ROUTE_LIST_ADAPTER, artifact="routes")


def save_routes(routes: list[Route], path: Path) -> None:
    _save_model(routes, path, _ROUTE_LIST_ADAPTER, artifact="routes")


def load_candidates(path: Path) -> list[Candidate]:
    return _load_model(path, _CANDIDATE_LIST_ADAPTER, artifact="candidates")


def save_candidates(candidates: list[Candidate], path: Path) -> None:
    _save_model(candidates, path, _CANDIDATE_LIST_ADAPTER, artifact="candidates")


def load_collected_routes(path: Path) -> CollectedRoutes:
    return _load_model(path, _COLLECTED_ROUTES_ADAPTER, artifact="collected_routes")


def save_collected_routes(routes: CollectedRoutes, path: Path) -> None:
    _save_model(routes, path, _COLLECTED_ROUTES_ADAPTER, artifact="collected_routes")


def load_collected_candidates(path: Path) -> CollectedCandidates:
    return _load_model(path, _COLLECTED_CANDIDATES_ADAPTER, artifact="collected_candidates")


def save_collected_candidates(candidates: CollectedCandidates, path: Path) -> None:
    _save_model(candidates, path, _COLLECTED_CANDIDATES_ADAPTER, artifact="collected_candidates")


def load_evaluation(path: Path) -> Evaluation:
    return _load_model(path, _EVALUATION_ADAPTER, artifact="evaluation")


def save_evaluation(evaluation: Evaluation, path: Path) -> None:
    _save_model(evaluation, path, _EVALUATION_ADAPTER, artifact="evaluation")


def load_analysis_report(path: Path) -> AnalysisReport:
    return _load_model(path, _ANALYSIS_REPORT_ADAPTER, artifact="analysis_report")


def save_analysis_report(report: AnalysisReport, path: Path) -> None:
    _save_model(report, path, _ANALYSIS_REPORT_ADAPTER, artifact="analysis_report")


@overload
def load_stock_file(path: Path, return_as: Literal["inchikey"] = "inchikey") -> set[InChIKeyStr]: ...


@overload
def load_stock_file(path: Path, return_as: Literal["smiles"]) -> set[SmilesStr]: ...


def load_stock_file(path: Path, return_as: Literal["inchikey", "smiles"] = "inchikey") -> set[InChIKeyStr] | set[SmilesStr]:
    if return_as not in ("inchikey", "smiles"):
        raise ValueError(f"invalid return_as parameter: {return_as!r}")
    if not path.exists():
        raise ArtifactNotFoundError(
            f"stock file not found: {path}",
            code="io.not_found",
            context={"path": str(path), "artifact": "stock"},
        )
    if not path.name.endswith(".csv.gz"):
        raise ArtifactFormatError(
            f"unsupported stock file format: {path}",
            code="io.unsupported_format",
            context={"path": str(path), "expected_suffix": ".csv.gz"},
        )

    required_column = "InChIKey" if return_as == "inchikey" else "SMILES"
    values = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or required_column not in reader.fieldnames:
                raise ArtifactFormatError(
                    f"invalid stock CSV format: missing {required_column}",
                    code="io.invalid_artifact_shape",
                    context={"path": str(path), "required_column": required_column, "columns": reader.fieldnames},
                )
            for row in reader:
                value = row.get(required_column, "").strip()
                if value:
                    values.add(InChIKeyStr(value) if return_as == "inchikey" else SmilesStr(value))
    except OSError as exc:
        raise ArtifactDecodeError(
            f"failed to read stock file {path}: {exc}",
            code="io.decode_failed",
            context={"path": str(path), "artifact": "stock"},
        ) from exc
    return values


def _load_model(path: Path, adapter: TypeAdapter[T], *, artifact: str) -> T:
    path = Path(path)
    try:
        return adapter.validate_python(load_json_gz(path))
    except ArtifactNotFoundError as exc:
        raise ArtifactNotFoundError(
            f"{artifact} file not found: {path}",
            code=exc.code,
            context={"path": str(path), "artifact": artifact},
        ) from exc
    except ArtifactDecodeError as exc:
        raise ArtifactDecodeError(
            f"Failed to load {artifact} from {path}: {exc}",
            code=exc.code,
            context={"path": str(path), "artifact": artifact},
        ) from exc
    except ValidationError as exc:
        raise ArtifactFormatError(
            f"Invalid {artifact} JSON format in {path}: {exc}",
            code="io.invalid_artifact_shape",
            context={"path": str(path), "artifact": artifact},
        ) from exc


def _save_model(value: T, path: Path, adapter: TypeAdapter[T], *, artifact: str) -> None:
    path = Path(path)
    try:
        payload = adapter.dump_python(
            value,
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        save_json_gz(payload, path)
    except ArtifactWriteError as exc:
        raise ArtifactWriteError(
            f"Failed to save {artifact} to {path}: {exc}",
            code=exc.code,
            context={"path": str(path), "artifact": artifact},
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactWriteError(
            f"Failed to save {artifact} to {path}: {exc}",
            code="io.write_failed",
            context={"path": str(path), "artifact": artifact},
        ) from exc
