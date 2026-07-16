from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path
from typing import Any, Literal, NoReturn

from retrocast.exceptions import (
    ArtifactFormatError,
    ConfigurationError,
    DatasetDownloadError,
    DatasetResolutionError,
    DatasetVerificationError,
)
from retrocast.hashing import hash_file
from retrocast.io import load_benchmark
from retrocast.models import StockTerminationConstraint
from retrocast.paths import validate_filename

TrainingDatasetName = Literal["paroutes"]
TrainingArtifactName = Literal[
    "n1-routes",
    "n5-routes",
    "route-holdout-n1-n5",
    "reaction-holdout-n1-n5",
    "n1-single-step-reactions",
    "n5-single-step-reactions",
    "single-step-route-holdout-n1-n5",
    "single-step-reaction-holdout-n1-n5",
]
TrainingSplitName = Literal["all", "training", "validation"]
TrainingSetFormat = Literal["jsonl", "rsmi"]
StockFormat = Literal["csv.gz", "txt.gz", "hdf5"]

DEFAULT_TRAINING_SET_BASE_URL = os.environ.get(
    "RETROCAST_TRAINING_SET_BASE_URL", "https://files.ischemist.com/retrocast/training-sets"
)
DEFAULT_TRAINING_SET_CACHE_SUBDIR = "training-sets"
DEFAULT_HOSTED_DATA_BASE_URL = os.environ.get(
    "RETROCAST_HOSTED_DATA_BASE_URL", "https://files.ischemist.com/retrocast/data"
)
DEFAULT_HOSTED_DATA_CACHE_SUBDIR = "data"
DEFAULT_DATASET_USER_AGENT = "retrocast/2"
SUPPORTED_DATASETS: tuple[TrainingDatasetName, ...] = ("paroutes",)
TRAINING_OMIT_PARTS = {"all", "training", "validation", "jsonl", "rsmi"}


def _raise_native_dataset_error(error: object) -> NoReturn:
    from retrocast import native

    assert isinstance(error, native.NativeDatasetError)
    payload = error.payload
    message = str(payload.get("message") or error)
    code = str(payload.get("code") or "dataset.error")
    context = payload.get("context")
    context = context if isinstance(context, dict) else {}
    retryable = bool(payload.get("retryable", False))
    category = str(payload.get("category") or "")
    error_type = {
        "configuration": ConfigurationError,
        "resolution": DatasetResolutionError,
        "download": DatasetDownloadError,
        "verification": DatasetVerificationError,
        "artifact_format": ArtifactFormatError,
    }.get(category, DatasetDownloadError)
    raise error_type(message, code=code, context=context, retryable=retryable) from error


@dataclass(frozen=True)
class DownloadedBenchmarkAssets:
    benchmark_path: Path
    stock_path: Path | None = None


@dataclass(frozen=True)
class PublishedFile:
    key: str
    sha256: str


@dataclass(frozen=True)
class TrainingArtifactSpec:
    supported_splits: tuple[TrainingSplitName, ...]
    supported_formats: tuple[TrainingSetFormat, ...]
    suffix_by_format: dict[TrainingSetFormat, str]


TRAINING_ARTIFACT_SPECS: dict[str, TrainingArtifactSpec] = {
    "n1-routes": TrainingArtifactSpec(
        supported_splits=("all",),
        supported_formats=("jsonl",),
        suffix_by_format={"jsonl": ".jsonl.gz"},
    ),
    "n5-routes": TrainingArtifactSpec(
        supported_splits=("all",),
        supported_formats=("jsonl",),
        suffix_by_format={"jsonl": ".jsonl.gz"},
    ),
    "route-holdout-n1-n5": TrainingArtifactSpec(
        supported_splits=("all", "training", "validation"),
        supported_formats=("jsonl",),
        suffix_by_format={"jsonl": ".jsonl.gz"},
    ),
    "reaction-holdout-n1-n5": TrainingArtifactSpec(
        supported_splits=("all", "training", "validation"),
        supported_formats=("jsonl",),
        suffix_by_format={"jsonl": ".jsonl.gz"},
    ),
    "n1-single-step-reactions": TrainingArtifactSpec(
        supported_splits=("all",),
        supported_formats=("jsonl", "rsmi"),
        suffix_by_format={"jsonl": ".jsonl.gz", "rsmi": ".rsmi.txt.gz"},
    ),
    "n5-single-step-reactions": TrainingArtifactSpec(
        supported_splits=("all",),
        supported_formats=("jsonl", "rsmi"),
        suffix_by_format={"jsonl": ".jsonl.gz", "rsmi": ".rsmi.txt.gz"},
    ),
    "single-step-route-holdout-n1-n5": TrainingArtifactSpec(
        supported_splits=("all", "training", "validation"),
        supported_formats=("jsonl", "rsmi"),
        suffix_by_format={"jsonl": ".jsonl.gz", "rsmi": ".rsmi.txt.gz"},
    ),
    "single-step-reaction-holdout-n1-n5": TrainingArtifactSpec(
        supported_splits=("all", "training", "validation"),
        supported_formats=("jsonl", "rsmi"),
        suffix_by_format={"jsonl": ".jsonl.gz", "rsmi": ".rsmi.txt.gz"},
    ),
}


def download_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    release: str = "latest",
    format: TrainingSetFormat = "jsonl",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
    show_progress: bool | None = None,
) -> Path:
    from retrocast import native

    try:
        return Path(
            native.dataset_download_training_set(
                {
                    "dataset": dataset,
                    "artifact": artifact,
                    "split": split,
                    "release": release,
                    "format": format,
                    "cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "output_dir": str(output_dir) if output_dir is not None else None,
                    "base_url": base_url,
                }
            )
        )
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def download_training_data(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName | None = None,
    split: TrainingSplitName | None = None,
    release: str = "latest",
    format: TrainingSetFormat | None = None,
    omit: tuple[str, ...] = (),
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
    dry_run: bool = False,
    show_progress: bool | None = None,
) -> list[Path]:
    from retrocast import native

    try:
        return [
            Path(path)
            for path in native.dataset_download_training_data(
                {
                    "dataset": dataset,
                    "artifact": artifact,
                    "split": split,
                    "release": release,
                    "format": format,
                    "omit": list(omit),
                    "cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "output_dir": str(output_dir) if output_dir is not None else None,
                    "base_url": base_url,
                    "dry_run": dry_run,
                }
            )
        ]
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def download_benchmark(
    name: str,
    *,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_HOSTED_DATA_BASE_URL,
    include_manifest: bool = False,
) -> Path:
    benchmark_name = validate_filename(name, "benchmark")
    benchmark_path = download_hosted_data_file(
        relative_path=Path("1-benchmarks") / "definitions" / f"{benchmark_name}.json.gz",
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_url=base_url,
    )
    if include_manifest:
        download_hosted_data_file(
            relative_path=Path("1-benchmarks") / "definitions" / f"{benchmark_name}.manifest.json",
            cache_dir=cache_dir,
            output_dir=output_dir,
            base_url=base_url,
        )
    return benchmark_path


def download_stock(
    name: str,
    *,
    format: StockFormat = "csv.gz",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_HOSTED_DATA_BASE_URL,
    include_manifest: bool = False,
) -> Path:
    stock_name = validate_filename(name, "stock")
    stock_format = validate_filename(format, "stock_format")
    stock_path = download_hosted_data_file(
        relative_path=Path("1-benchmarks") / "stocks" / f"{stock_name}.{stock_format}",
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_url=base_url,
    )
    if include_manifest:
        download_hosted_data_file(
            relative_path=Path("1-benchmarks") / "stocks" / f"{stock_name}.manifest.json",
            cache_dir=cache_dir,
            output_dir=output_dir,
            base_url=base_url,
        )
    return stock_path


def download_benchmark_assets(
    name: str,
    *,
    stock_format: StockFormat = "csv.gz",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_HOSTED_DATA_BASE_URL,
    include_manifests: bool = False,
) -> DownloadedBenchmarkAssets:
    benchmark_path = download_benchmark(
        name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_url=base_url,
        include_manifest=include_manifests,
    )
    stock_name = _single_stock_name(load_benchmark(benchmark_path))
    stock_path = (
        download_stock(
            stock_name,
            format=stock_format,
            cache_dir=cache_dir,
            output_dir=output_dir,
            base_url=base_url,
            include_manifest=include_manifests,
        )
        if stock_name is not None
        else None
    )
    return DownloadedBenchmarkAssets(benchmark_path=benchmark_path, stock_path=stock_path)


def download_hosted_data_target(
    target: str,
    *,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_HOSTED_DATA_BASE_URL,
    dry_run: bool = False,
) -> list[Path]:
    from retrocast import native

    try:
        return [
            Path(path)
            for path in native.dataset_download_hosted_data(
                {
                    "target": target,
                    "cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "output_dir": str(output_dir) if output_dir is not None else None,
                    "base_url": base_url,
                    "dry_run": dry_run,
                }
            )
        ]
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def hosted_data_target_selector(target: str) -> Callable[[str], bool]:
    benchmark_dependencies = {
        "mkt-lin-500": ("mkt-lin-500.", "buyables-stock"),
        "mkt-cnv-160": ("mkt-cnv-160.", "buyables-stock"),
        "mkt-cnv-160-depth": ("mkt-cnv-160-depth.", "buyables-stock"),
        "mkt-cnv-160-leaf": ("mkt-cnv-160-leaf.", "buyables-stock"),
        "mkt-cnv-160-leaf-depth": ("mkt-cnv-160-leaf-depth.", "buyables-stock"),
        "ref-lin-600": ("ref-lin-600", "n5-stock"),
        "ref-cnv-400": ("ref-cnv-400", "n5-stock"),
        "ref-lng-84": ("ref-lng-84", "n1-n5-stock"),
    }
    selectors: dict[str, Callable[[str], bool]] = {
        "all": lambda key: True,
        "benchmarks": lambda key: key.startswith("1-benchmarks"),
        "definitions": lambda key: key.startswith("1-benchmarks/definitions"),
        "stocks": lambda key: key.startswith("1-benchmarks/stocks"),
        "raw": lambda key: key.startswith("2-raw"),
        "processed": lambda key: key.startswith("3-processed"),
        "scored": lambda key: key.startswith("4-scored"),
        "results": lambda key: key.startswith("5-results"),
    }
    for name, parts in benchmark_dependencies.items():
        selectors[name] = lambda key, required=parts: (
            key.startswith("1-benchmarks/") and any(part in key for part in required)
        )
    try:
        return selectors[target]
    except KeyError as exc:
        raise ConfigurationError(
            f"unsupported hosted data target: {target}",
            code="dataset.unsupported_target",
            context={"target": target, "supported_targets": sorted(selectors)},
        ) from exc


def validate_training_dataset_request(
    *,
    dataset: str,
    artifact: str,
    split: str,
    format: str,
) -> None:
    from retrocast import native

    try:
        native.dataset_validate_training_request(dataset, artifact, split, format)
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def validate_training_omit_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if part not in TRAINING_OMIT_PARTS:
            raise ConfigurationError(
                f"unsupported omit part: {part}",
                code="dataset.unsupported_omit_part",
                context={"omit_part": part, "supported_omit_parts": sorted(TRAINING_OMIT_PARTS)},
            )


def published_training_files(
    *,
    checksums_path: Path,
    checksums_url: str,
    artifact: str | None,
    split: str | None,
    format: str | None,
    omit: tuple[str, ...],
) -> list[PublishedFile]:
    hashes = download_sha256sums(checksums_url, checksums_path)
    return [
        PublishedFile(key=key, sha256=sha256)
        for key, sha256 in hashes.items()
        if training_file_matches(key=key, artifact=artifact, split=split, format=format, omit=omit)
    ]


def published_hosted_data_files(checksums_path: Path, checksums_url: str) -> list[PublishedFile]:
    hashes = download_sha256sums(checksums_url, checksums_path)
    return [PublishedFile(key=key, sha256=sha256) for key, sha256 in hashes.items()]


def training_file_matches(
    *,
    key: str,
    artifact: str | None,
    split: str | None,
    format: str | None,
    omit: tuple[str, ...],
) -> bool:
    from retrocast import native

    return native.dataset_training_file_matches(
        key,
        artifact=artifact,
        split=split,
        format=format,
        omit=omit,
    )


def training_file_part_matches(filename: str, part: str) -> bool:
    return filename == f"{part}.jsonl.gz" or filename == f"{part}.rsmi.txt.gz" or f".{part}." in filename


def resolve_latest_training_set_release(
    dataset: TrainingDatasetName,
    *,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> str:
    return resolve_training_set_release(dataset=dataset, release="latest", base_url=base_url)


def resolve_training_set_release(
    *,
    dataset: str,
    release: str = "latest",
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> str:
    from retrocast import native

    try:
        return native.dataset_resolve_release(dataset, release, base_url)
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def resolve_training_set_filename(*, artifact: TrainingArtifactName, split: str, format: TrainingSetFormat) -> str:
    from retrocast import native

    try:
        return native.dataset_training_filename(artifact, split, format)
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def resolve_training_set_root(
    *,
    dataset: str,
    release: str,
    cache_dir: Path | None,
    output_dir: Path | None,
) -> Path:
    from retrocast import native

    return Path(
        native.dataset_training_root(
            dataset,
            release,
            str(cache_dir) if cache_dir is not None else None,
            str(output_dir) if output_dir is not None else None,
        )
    )


def resolve_hosted_data_root(*, cache_dir: Path | None, output_dir: Path | None) -> Path:
    from retrocast import native

    return Path(
        native.dataset_hosted_root(
            str(cache_dir) if cache_dir is not None else None,
            str(output_dir) if output_dir is not None else None,
        )
    )


def build_training_set_url(
    *,
    base_url: str,
    dataset: str,
    filename: str,
    release: str | None = None,
    artifact: str | None = None,
) -> str:
    from retrocast import native

    parts = [dataset]
    if release is not None:
        parts.append(release)
    if artifact is not None:
        parts.append(artifact)
    parts.append(filename)
    return native.dataset_build_url(base_url, parts)


def build_training_set_checksum_key(*, artifact: str, filename: str) -> str:
    return f"{artifact}/{filename}"


def build_training_data_file_url(*, base_url: str, dataset: str, release: str, relative_path: Path) -> str:
    from retrocast import native

    return native.dataset_build_url(base_url, [dataset, release, *relative_path.parts])


def build_hosted_data_url(*, base_url: str, relative_path: Path) -> str:
    from retrocast import native

    return native.dataset_build_url(base_url, relative_path.parts)


def should_show_download_progress(show_progress: bool | None) -> bool:
    if show_progress is not None:
        return show_progress
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except (AttributeError, OSError):
        return False


def download_hosted_data_file(
    *,
    relative_path: Path,
    cache_dir: Path | None,
    output_dir: Path | None,
    base_url: str,
) -> Path:
    from retrocast import native

    try:
        return Path(
            native.dataset_download_hosted_file(
                {
                    "relative_path": str(relative_path),
                    "cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "output_dir": str(output_dir) if output_dir is not None else None,
                    "base_url": base_url,
                }
            )
        )
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def download_verified_file(
    *,
    local_path: Path,
    checksums_path: Path,
    checksums_url: str,
    checksum_key: str,
    download_url: str,
    missing_message: str,
    missing_context: dict[str, object],
    mismatch_message: str,
    mismatch_context: dict[str, object],
    show_progress: bool = False,
    progress_description: str | None = None,
) -> str:
    expected_hash = resolve_expected_hash(
        checksums_path=checksums_path,
        checksums_url=checksums_url,
        checksum_key=checksum_key,
        missing_message=missing_message,
        missing_context=missing_context,
    )
    if local_path.exists() and hash_file(local_path) == expected_hash:
        return expected_hash

    download_url_to_path(
        download_url,
        local_path,
        show_progress=show_progress,
        progress_description=progress_description,
    )
    actual_hash = hash_file(local_path)
    if actual_hash != expected_hash:
        local_path.unlink(missing_ok=True)
        checksums_path.unlink(missing_ok=True)
        raise DatasetVerificationError(
            mismatch_message,
            code="dataset.hash_mismatch",
            context={**mismatch_context, "expected_sha256": expected_hash, "actual_sha256": actual_hash},
        )
    return expected_hash


def resolve_expected_hash(
    *,
    checksums_path: Path,
    checksums_url: str,
    checksum_key: str,
    missing_message: str,
    missing_context: dict[str, object],
) -> str:
    from retrocast import native

    del missing_message, missing_context
    try:
        return native.dataset_resolve_expected(str(checksums_path), checksums_url, checksum_key)
    except native.NativeDatasetError as error:
        _raise_native_dataset_error(error)


def load_or_download_checksums(*, checksums_path: Path, checksums_url: str) -> dict[str, str]:
    return (
        load_sha256sums(checksums_path)
        if checksums_path.exists()
        else download_sha256sums(checksums_url, checksums_path)
    )


def download_sha256sums(url: str, destination: Path) -> dict[str, str]:
    download_url_to_path(url, destination)
    return load_sha256sums(destination)


def load_sha256sums(path: Path) -> dict[str, str]:
    from retrocast import native

    try:
        return native.dataset_load_sha256sums(str(path))
    except native.NativeDatasetError as exc:
        if exc.payload.get("kind") == "invalid_checksum":
            line = exc.payload.get("line", "")
            raise ArtifactFormatError(
                f"invalid dataset checksum line: {line!r}",
                code="dataset.invalid_checksums",
                context={"path": str(path), "line": line},
            ) from exc
        raise DatasetDownloadError(
            f"failed to read dataset checksums from {path}",
            code="dataset.checksums_read_failed",
            context={"path": str(path)},
        ) from exc


def load_json_url(url: str) -> object:
    from retrocast import native

    try:
        return native.dataset_load_json_url(url)
    except native.NativeDatasetError as exc:
        kind = exc.payload.get("kind")
        if kind == "http_status":
            status = exc.payload.get("status")
            raise DatasetResolutionError(
                f"failed to resolve hosted metadata at {url}: HTTP {status}",
                code="dataset.metadata_http_error",
                context={"url": url, "status": status},
            ) from exc
        if kind == "invalid_json":
            raise ArtifactFormatError(
                f"invalid json returned by hosted metadata endpoint {url}",
                code="dataset.invalid_metadata_json",
                context={"url": url},
            ) from exc
        raise DatasetDownloadError(
            f"failed to reach hosted metadata at {url}: {exc}",
            code="dataset.metadata_unreachable",
            context={"url": url},
            retryable=True,
        ) from exc


def download_url_to_path(
    url: str,
    destination: Path,
    *,
    show_progress: bool = False,
    progress_description: str | None = None,
) -> None:
    from retrocast import native

    try:
        native.dataset_download_url_to_path(url, str(destination))
    except native.NativeDatasetError as exc:
        kind = exc.payload.get("kind")
        if kind == "http_status":
            status = exc.payload.get("status")
            raise DatasetResolutionError(
                f"failed to download hosted file from {url}: HTTP {status}",
                code="dataset.file_http_error",
                context={"url": url, "status": status},
            ) from exc
        if kind == "write":
            raise DatasetDownloadError(
                f"failed to write downloaded hosted file to {destination}",
                code="dataset.cache_write_failed",
                context={"path": str(destination), "url": url},
            ) from exc
        raise DatasetDownloadError(
            f"failed to reach hosted file at {url}: {exc}",
            code="dataset.file_unreachable",
            context={"url": url},
            retryable=True,
        ) from exc


def write_response_with_progress(*, response: Any, handle: BufferedWriter, description: str) -> None:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    content_length = response.headers.get("Content-Length")
    try:
        total_bytes = int(content_length) if content_length is not None else None
    except ValueError:
        total_bytes = None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=total_bytes)
        while chunk := response.read(8192):
            handle.write(chunk)
            progress.update(task_id, advance=len(chunk))


def _single_stock_name(benchmark: Any) -> str | None:
    stock_names = set()
    for target_id in benchmark.targets:
        for constraint in benchmark.effective_constraints(target_id):
            if isinstance(constraint, StockTerminationConstraint):
                stock_names.add(constraint.stock)
    if len(stock_names) == 1:
        return next(iter(stock_names))
    return None
