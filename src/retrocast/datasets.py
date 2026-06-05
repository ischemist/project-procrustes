from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

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
from retrocast.paths import resolve_cache_dir, validate_filename

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


class LatestTrainingSetRelease(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    dataset: str
    latest_release: str


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
    validate_training_dataset_request(dataset=dataset, artifact=artifact, split=split, format=format)
    resolved_release = resolve_training_set_release(dataset=dataset, release=release, base_url=base_url)
    release_dir = resolve_training_set_root(
        dataset=dataset,
        release=resolved_release,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    filename = resolve_training_set_filename(artifact=artifact, split=split, format=format)
    checksums_path = release_dir / "SHA256SUMS"
    checksums_url = build_training_set_url(
        base_url=base_url,
        dataset=dataset,
        release=resolved_release,
        filename="SHA256SUMS",
    )
    artifact_dir = release_dir / artifact
    artifact_path = artifact_dir / filename

    download_verified_file(
        local_path=artifact_path,
        checksums_path=checksums_path,
        checksums_url=checksums_url,
        checksum_key=build_training_set_checksum_key(artifact=artifact, filename=filename),
        download_url=build_training_set_url(
            base_url=base_url,
            dataset=dataset,
            release=resolved_release,
            artifact=artifact,
            filename=filename,
        ),
        show_progress=should_show_download_progress(show_progress),
        progress_description=f"downloading {artifact}/{filename}",
        missing_message=f"artifact '{artifact}' release '{resolved_release}' does not publish '{filename}'",
        missing_context={"dataset": dataset, "artifact": artifact, "release": resolved_release, "filename": filename},
        mismatch_message=f"downloaded dataset file failed integrity verification: {filename}",
        mismatch_context={"dataset": dataset, "artifact": artifact, "release": resolved_release, "filename": filename},
    )
    download_verified_file(
        local_path=artifact_dir / "manifest.json",
        checksums_path=checksums_path,
        checksums_url=checksums_url,
        checksum_key=build_training_set_checksum_key(artifact=artifact, filename="manifest.json"),
        download_url=build_training_set_url(
            base_url=base_url,
            dataset=dataset,
            release=resolved_release,
            artifact=artifact,
            filename="manifest.json",
        ),
        missing_message=f"artifact '{artifact}' release '{resolved_release}' does not publish 'manifest.json'",
        missing_context={
            "dataset": dataset,
            "artifact": artifact,
            "release": resolved_release,
            "filename": "manifest.json",
        },
        mismatch_message="downloaded dataset manifest failed integrity verification",
        mismatch_context={"dataset": dataset, "artifact": artifact, "release": resolved_release},
    )
    return artifact_path


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
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigurationError(
            f"unsupported training dataset: {dataset}",
            code="dataset.unsupported_dataset",
            context={"dataset": dataset, "supported_datasets": list(SUPPORTED_DATASETS)},
        )
    if artifact is not None and artifact not in TRAINING_ARTIFACT_SPECS:
        raise ConfigurationError(
            f"unsupported training artifact: {artifact}",
            code="dataset.unsupported_artifact",
            context={"artifact": artifact},
        )
    if split is not None and split not in {"all", "training", "validation"}:
        raise ConfigurationError(
            f"unsupported training split: {split}",
            code="dataset.unsupported_split",
            context={"split": split},
        )
    if format is not None and format not in {"jsonl", "rsmi"}:
        raise ConfigurationError(
            f"unsupported training dataset format: {format}",
            code="dataset.unsupported_format",
            context={"format": format},
        )
    validate_training_omit_parts(omit)

    resolved_release = resolve_training_set_release(dataset=dataset, release=release, base_url=base_url)
    release_dir = resolve_training_set_root(
        dataset=dataset,
        release=resolved_release,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    checksums_path = release_dir / "SHA256SUMS"
    checksums_url = build_training_set_url(
        base_url=base_url,
        dataset=dataset,
        release=resolved_release,
        filename="SHA256SUMS",
    )
    published_files = published_training_files(
        checksums_path=checksums_path,
        checksums_url=checksums_url,
        artifact=artifact,
        split=split,
        format=format,
        omit=omit,
    )
    if not published_files:
        raise DatasetResolutionError(
            "no published training data files match request",
            code="dataset.no_matching_files",
            context={
                "dataset": dataset,
                "release": resolved_release,
                "artifact": artifact,
                "split": split,
                "format": format,
                "omit": list(omit),
            },
        )

    paths = [release_dir / file.key for file in published_files]
    if dry_run:
        return paths

    for file, path in zip(published_files, paths, strict=True):
        download_verified_file(
            local_path=path,
            checksums_path=checksums_path,
            checksums_url=checksums_url,
            checksum_key=file.key,
            download_url=build_training_data_file_url(
                base_url=base_url,
                dataset=dataset,
                release=resolved_release,
                relative_path=Path(file.key),
            ),
            show_progress=should_show_download_progress(show_progress),
            progress_description=f"downloading {file.key}",
            missing_message=f"training data file is not published: {file.key}",
            missing_context={"dataset": dataset, "release": resolved_release, "key": file.key},
            mismatch_message=f"downloaded training data file failed integrity verification: {file.key}",
            mismatch_context={"dataset": dataset, "release": resolved_release, "key": file.key},
        )
    return paths


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
    selector = hosted_data_target_selector(target)
    root_dir = resolve_hosted_data_root(cache_dir=cache_dir, output_dir=output_dir)
    checksums_path = root_dir / "SHA256SUMS"
    checksums_url = build_hosted_data_url(base_url=base_url, relative_path=Path("SHA256SUMS"))
    files = [file for file in published_hosted_data_files(checksums_path, checksums_url) if selector(file.key)]
    if not files:
        raise DatasetResolutionError(
            f"no hosted data files match target: {target}",
            code="dataset.no_matching_files",
            context={"target": target},
        )
    paths = [root_dir / file.key for file in files]
    if dry_run:
        return paths
    for file, path in zip(files, paths, strict=True):
        download_verified_file(
            local_path=path,
            checksums_path=checksums_path,
            checksums_url=checksums_url,
            checksum_key=file.key,
            download_url=build_hosted_data_url(base_url=base_url, relative_path=Path(file.key)),
            missing_message=f"hosted data file is not published: {file.key}",
            missing_context={"target": target, "relative_path": file.key},
            mismatch_message=f"downloaded hosted data file failed integrity verification: {Path(file.key).name}",
            mismatch_context={"target": target, "relative_path": file.key},
        )
    return paths


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
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigurationError(
            f"unsupported training dataset: {dataset}",
            code="dataset.unsupported_dataset",
            context={"dataset": dataset, "supported_datasets": list(SUPPORTED_DATASETS)},
        )
    if artifact not in TRAINING_ARTIFACT_SPECS:
        raise ConfigurationError(
            f"unsupported training artifact: {artifact}",
            code="dataset.unsupported_artifact",
            context={"artifact": artifact},
        )
    if split not in {"all", "training", "validation"}:
        raise ConfigurationError(
            f"unsupported training split: {split}",
            code="dataset.unsupported_split",
            context={"split": split},
        )
    artifact_spec = TRAINING_ARTIFACT_SPECS[artifact]
    if split not in artifact_spec.supported_splits:
        raise ConfigurationError(
            f"artifact '{artifact}' does not support split '{split}'",
            code="dataset.split_mismatch",
            context={"artifact": artifact, "split": split, "supported_splits": list(artifact_spec.supported_splits)},
        )
    if format not in {"jsonl", "rsmi"}:
        raise ConfigurationError(
            f"unsupported training dataset format: {format}",
            code="dataset.unsupported_format",
            context={"format": format},
        )
    supported_formats = artifact_spec.supported_formats
    if format not in supported_formats:
        raise ConfigurationError(
            f"artifact '{artifact}' does not support format '{format}'",
            code="dataset.format_mismatch",
            context={"artifact": artifact, "format": format, "supported_formats": list(supported_formats)},
        )


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
    hashes = load_or_download_checksums(checksums_path=checksums_path, checksums_url=checksums_url)
    return [
        PublishedFile(key=key, sha256=sha256)
        for key, sha256 in hashes.items()
        if training_file_matches(key=key, artifact=artifact, split=split, format=format, omit=omit)
    ]


def published_hosted_data_files(checksums_path: Path, checksums_url: str) -> list[PublishedFile]:
    hashes = load_or_download_checksums(checksums_path=checksums_path, checksums_url=checksums_url)
    return [PublishedFile(key=key, sha256=sha256) for key, sha256 in hashes.items()]


def training_file_matches(
    *,
    key: str,
    artifact: str | None,
    split: str | None,
    format: str | None,
    omit: tuple[str, ...],
) -> bool:
    if artifact is not None and not key.startswith(f"{artifact}/"):
        return False
    filename = Path(key).name
    if filename == "manifest.json":
        return artifact is not None or (split is None and format is None)
    if split is not None and not training_file_part_matches(filename, split):
        return False
    if format == "jsonl" and not filename.endswith(".jsonl.gz"):
        return False
    if format == "rsmi" and not filename.endswith(".rsmi.txt.gz"):
        return False
    return not any(training_file_part_matches(filename, part) for part in omit)


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
    if release != "latest":
        return release

    latest_url = build_training_set_url(base_url=base_url, dataset=dataset, filename="latest.json")
    payload = load_json_url(latest_url)
    try:
        latest = LatestTrainingSetRelease.model_validate(payload)
    except ValidationError as exc:
        raise ArtifactFormatError(
            f"invalid latest release payload for training dataset '{dataset}'",
            code="dataset.invalid_latest_payload",
            context={"dataset": dataset, "url": latest_url},
        ) from exc
    if latest.dataset != dataset:
        raise DatasetResolutionError(
            f"latest release pointer dataset mismatch: expected '{dataset}', got '{latest.dataset}'",
            code="dataset.latest_dataset_mismatch",
            context={"dataset": dataset, "resolved_dataset": latest.dataset, "url": latest_url},
        )
    return latest.latest_release


def resolve_training_set_filename(*, artifact: TrainingArtifactName, split: str, format: TrainingSetFormat) -> str:
    suffix = TRAINING_ARTIFACT_SPECS[artifact].suffix_by_format[format]
    return f"{split}{suffix}"


def resolve_training_set_root(
    *,
    dataset: str,
    release: str,
    cache_dir: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir) / release
    if cache_dir is not None:
        return Path(cache_dir) / dataset / release
    return resolve_cache_dir(DEFAULT_TRAINING_SET_CACHE_SUBDIR) / dataset / release


def resolve_hosted_data_root(*, cache_dir: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if cache_dir is not None:
        return Path(cache_dir)
    return resolve_cache_dir(DEFAULT_HOSTED_DATA_CACHE_SUBDIR)


def build_training_set_url(
    *,
    base_url: str,
    dataset: str,
    filename: str,
    release: str | None = None,
    artifact: str | None = None,
) -> str:
    parts = [base_url.rstrip("/"), quote(dataset, safe="")]
    if release is not None:
        parts.append(quote(release, safe=""))
    if artifact is not None:
        parts.append(quote(artifact, safe=""))
    parts.append(quote(filename, safe="."))
    return "/".join(parts)


def build_training_set_checksum_key(*, artifact: str, filename: str) -> str:
    return f"{artifact}/{filename}"


def build_training_data_file_url(*, base_url: str, dataset: str, release: str, relative_path: Path) -> str:
    return "/".join(
        [
            base_url.rstrip("/"),
            quote(dataset, safe=""),
            quote(release, safe=""),
            *(quote(part, safe=".") for part in relative_path.parts),
        ]
    )


def build_hosted_data_url(*, base_url: str, relative_path: Path) -> str:
    return "/".join([base_url.rstrip("/"), *(quote(part, safe=".") for part in relative_path.parts)])


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
    root_dir = resolve_hosted_data_root(cache_dir=cache_dir, output_dir=output_dir)
    local_path = root_dir / relative_path
    checksums_path = root_dir / "SHA256SUMS"
    checksums_url = build_hosted_data_url(base_url=base_url, relative_path=Path("SHA256SUMS"))
    download_verified_file(
        local_path=local_path,
        checksums_path=checksums_path,
        checksums_url=checksums_url,
        checksum_key=str(relative_path),
        download_url=build_hosted_data_url(base_url=base_url, relative_path=relative_path),
        missing_message=f"hosted data file is not published: {relative_path}",
        missing_context={"relative_path": str(relative_path), "checksums_url": checksums_url},
        mismatch_message=f"downloaded hosted data file failed integrity verification: {relative_path.name}",
        mismatch_context={"relative_path": str(relative_path)},
    )
    return local_path


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
    hashes = load_or_download_checksums(checksums_path=checksums_path, checksums_url=checksums_url)
    expected_hash = hashes.get(checksum_key)
    if expected_hash is not None:
        return expected_hash

    hashes = download_sha256sums(checksums_url, checksums_path)
    expected_hash = hashes.get(checksum_key)
    if expected_hash is None:
        raise DatasetResolutionError(missing_message, code="dataset.file_not_published", context=missing_context)
    return expected_hash


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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetDownloadError(
            f"failed to read dataset checksums from {path}",
            code="dataset.checksums_read_failed",
            context={"path": str(path)},
        ) from exc

    checksums = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            sha256, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise ArtifactFormatError(
                f"invalid dataset checksum line: {line!r}",
                code="dataset.invalid_checksums",
                context={"path": str(path), "line": line},
            ) from exc
        checksums[filename.strip()] = sha256
    return checksums


def load_json_url(url: str) -> object:
    try:
        with urlopen(Request(url, headers={"User-Agent": DEFAULT_DATASET_USER_AGENT})) as response:
            return json.load(response)
    except HTTPError as exc:
        raise DatasetResolutionError(
            f"failed to resolve hosted metadata at {url}: HTTP {exc.code}",
            code="dataset.metadata_http_error",
            context={"url": url, "status": exc.code},
        ) from exc
    except URLError as exc:
        raise DatasetDownloadError(
            f"failed to reach hosted metadata at {url}: {exc.reason}",
            code="dataset.metadata_unreachable",
            context={"url": url},
            retryable=True,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactFormatError(
            f"invalid json returned by hosted metadata endpoint {url}",
            code="dataset.invalid_metadata_json",
            context={"url": url},
        ) from exc


def download_url_to_path(
    url: str,
    destination: Path,
    *,
    show_progress: bool = False,
    progress_description: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with (
            urlopen(Request(url, headers={"User-Agent": DEFAULT_DATASET_USER_AGENT})) as response,
            tmp_path.open("wb") as handle,
        ):
            if show_progress:
                write_response_with_progress(
                    response=response,
                    handle=handle,
                    description=progress_description or f"downloading {destination.name}",
                )
            else:
                while chunk := response.read(8192):
                    handle.write(chunk)
    except HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        raise DatasetResolutionError(
            f"failed to download hosted file from {url}: HTTP {exc.code}",
            code="dataset.file_http_error",
            context={"url": url, "status": exc.code},
        ) from exc
    except URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"failed to reach hosted file at {url}: {exc.reason}",
            code="dataset.file_unreachable",
            context={"url": url},
            retryable=True,
        ) from exc
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"failed to write downloaded hosted file to {destination}",
            code="dataset.cache_write_failed",
            context={"path": str(destination), "url": url},
        ) from exc
    try:
        tmp_path.replace(destination)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"failed to write downloaded hosted file to {destination}",
            code="dataset.cache_write_failed",
            context={"path": str(destination), "url": url},
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
