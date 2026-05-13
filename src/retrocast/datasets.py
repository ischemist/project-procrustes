from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from retrocast.curation.training.records import TrainingReactionRecord, TrainingRouteRecord
from retrocast.exceptions import (
    ArtifactFormatError,
    ConfigurationError,
    DatasetDownloadError,
    DatasetResolutionError,
    DatasetVerificationError,
)
from retrocast.io import (
    load_benchmark,
    load_training_reaction_records,
    load_training_reaction_smiles,
    load_training_route_records,
    load_training_routes,
)
from retrocast.paths import validate_filename

if TYPE_CHECKING:
    from retrocast.models.chem import Route

TrainingDatasetName = Literal["paroutes"]
TrainingArtifactName = Literal[
    "route-holdout-n1-n5",
    "reaction-holdout-n1-n5",
    "single-step-reaction-holdout-n1-n5",
]
TrainingSplitName = Literal["all", "training", "validation"]
TrainingSetFormat = Literal["routes", "route_records", "reaction_records", "reaction_smiles"]
StockFormat = Literal["csv.gz", "txt.gz", "hdf5"]

DEFAULT_TRAINING_SET_BASE_URL = os.getenv(
    "RETROCAST_TRAINING_SET_BASE_URL",
    "https://files.ischemist.com/retrocast/training-sets",
)
DEFAULT_TRAINING_SET_CACHE_DIR = Path(
    os.getenv(
        "RETROCAST_TRAINING_SET_CACHE_DIR",
        str(Path.home() / ".cache" / "retrocast" / "training-sets"),
    )
)
DEFAULT_HOSTED_DATA_BASE_URL = os.getenv(
    "RETROCAST_HOSTED_DATA_BASE_URL",
    "https://files.ischemist.com/retrocast/data",
)
DEFAULT_HOSTED_DATA_CACHE_DIR = Path(
    os.getenv(
        "RETROCAST_HOSTED_DATA_CACHE_DIR",
        str(Path.home() / ".cache" / "retrocast" / "data"),
    )
)
DEFAULT_DATASET_USER_AGENT = "retrocast/1.0"
SUPPORTED_DATASETS: tuple[TrainingDatasetName, ...] = ("paroutes",)
ROUTE_ARTIFACTS = {"route-holdout-n1-n5", "reaction-holdout-n1-n5"}
SINGLE_STEP_ARTIFACT = "single-step-reaction-holdout-n1-n5"


class LatestTrainingSetRelease(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    dataset: str
    latest_release: str


@dataclass(frozen=True)
class DownloadedBenchmarkAssets:
    benchmark_path: Path
    stock_path: Path | None = None


@dataclass(frozen=True)
class DownloadedTrainingSet:
    dataset: TrainingDatasetName
    artifact: TrainingArtifactName
    split: TrainingSplitName
    format: TrainingSetFormat
    requested_release: str
    resolved_release: str
    filename: str
    path: Path
    checksums_path: Path
    sha256: str


def download_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    release: str = "latest",
    as_: TrainingSetFormat,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> Path:
    return download_training_set_info(
        dataset,
        artifact=artifact,
        split=split,
        release=release,
        as_=as_,
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_url=base_url,
    ).path


def download_training_set_info(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    release: str = "latest",
    as_: TrainingSetFormat,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> DownloadedTrainingSet:
    validate_training_dataset_request(dataset=dataset, artifact=artifact, split=split, as_=as_)
    resolved_release = resolve_training_set_release(dataset=dataset, release=release, base_url=base_url)
    filename = resolve_training_set_filename(artifact=artifact, split=split, as_=as_)
    release_cache_dir = resolve_training_set_root(
        dataset=dataset,
        release=resolved_release,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    artifact_cache_dir = release_cache_dir / artifact
    artifact_cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = artifact_cache_dir / filename
    checksums_path = release_cache_dir / "SHA256SUMS"
    checksums_url = build_training_set_url(
        base_url=base_url,
        dataset=dataset,
        release=resolved_release,
        filename="SHA256SUMS",
    )
    checksum_key = build_training_set_checksum_key(artifact=artifact, filename=filename)

    expected_hashes = (
        load_sha256sums(checksums_path)
        if checksums_path.exists()
        else download_sha256sums(checksums_url, checksums_path)
    )
    expected_hash = expected_hashes.get(checksum_key)
    if expected_hash is None:
        expected_hashes = download_sha256sums(checksums_url, checksums_path)
        expected_hash = expected_hashes.get(checksum_key)
    if expected_hash is None:
        raise DatasetResolutionError(
            f"artifact '{artifact}' release '{resolved_release}' does not publish '{filename}'",
            code="dataset.file_not_published",
            context={
                "dataset": dataset,
                "artifact": artifact,
                "release": resolved_release,
                "filename": filename,
                "checksum_key": checksum_key,
                "checksums_url": checksums_url,
            },
        )

    if local_path.exists() and sha256_file(local_path) == expected_hash:
        return DownloadedTrainingSet(
            dataset=dataset,
            artifact=artifact,
            split=split,
            format=as_,
            requested_release=release,
            resolved_release=resolved_release,
            filename=filename,
            path=local_path,
            checksums_path=checksums_path,
            sha256=expected_hash,
        )

    download_url = build_training_set_url(
        base_url=base_url,
        dataset=dataset,
        release=resolved_release,
        artifact=artifact,
        filename=filename,
    )
    download_url_to_path(download_url, local_path)

    actual_hash = sha256_file(local_path)
    if actual_hash != expected_hash:
        invalidate_cached_download(local_path=local_path, checksums_path=checksums_path)
        raise DatasetVerificationError(
            f"downloaded dataset file failed integrity verification: {filename}",
            code="dataset.hash_mismatch",
            context={
                "dataset": dataset,
                "artifact": artifact,
                "release": resolved_release,
                "filename": filename,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            },
        )
    return DownloadedTrainingSet(
        dataset=dataset,
        artifact=artifact,
        split=split,
        format=as_,
        requested_release=release,
        resolved_release=resolved_release,
        filename=filename,
        path=local_path,
        checksums_path=checksums_path,
        sha256=expected_hash,
    )


@overload
def load_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    as_: Literal["routes"],
    release: str = "latest",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> list[Route]: ...


@overload
def load_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    as_: Literal["route_records"],
    release: str = "latest",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> list[TrainingRouteRecord]: ...


@overload
def load_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    as_: Literal["reaction_records"],
    release: str = "latest",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> list[TrainingReactionRecord]: ...


@overload
def load_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    as_: Literal["reaction_smiles"],
    release: str = "latest",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> list[str]: ...


def load_training_set(
    dataset: TrainingDatasetName,
    *,
    artifact: TrainingArtifactName,
    split: TrainingSplitName,
    as_: TrainingSetFormat,
    release: str = "latest",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    base_url: str = DEFAULT_TRAINING_SET_BASE_URL,
) -> list[Route] | list[TrainingRouteRecord] | list[TrainingReactionRecord] | list[str]:
    path = download_training_set(
        dataset,
        artifact=artifact,
        split=split,
        release=release,
        as_=as_,
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_url=base_url,
    )
    if as_ == "routes":
        return load_training_routes(path)
    if as_ == "route_records":
        return load_training_route_records(path)
    if as_ == "reaction_records":
        return load_training_reaction_records(path)
    return load_training_reaction_smiles(path)


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
    benchmark = load_benchmark(benchmark_path)
    stock_path = (
        download_stock(
            benchmark.stock_name,
            format=stock_format,
            cache_dir=cache_dir,
            output_dir=output_dir,
            base_url=base_url,
            include_manifest=include_manifests,
        )
        if benchmark.stock_name
        else None
    )
    return DownloadedBenchmarkAssets(benchmark_path=benchmark_path, stock_path=stock_path)


def validate_training_dataset_request(
    *,
    dataset: str,
    artifact: str,
    split: str,
    as_: str,
) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigurationError(
            f"unsupported training dataset: {dataset}",
            code="dataset.unsupported_dataset",
            context={"dataset": dataset, "supported_datasets": list(SUPPORTED_DATASETS)},
        )
    if artifact not in ROUTE_ARTIFACTS | {SINGLE_STEP_ARTIFACT}:
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
    if as_ not in {"routes", "route_records", "reaction_records", "reaction_smiles"}:
        raise ConfigurationError(
            f"unsupported training dataset format: {as_}",
            code="dataset.unsupported_format",
            context={"format": as_},
        )
    if artifact in ROUTE_ARTIFACTS and as_ not in {"routes", "route_records"}:
        raise ConfigurationError(
            f"artifact '{artifact}' does not support format '{as_}'",
            code="dataset.format_mismatch",
            context={"artifact": artifact, "format": as_},
        )
    if artifact == SINGLE_STEP_ARTIFACT and as_ not in {"reaction_records", "reaction_smiles"}:
        raise ConfigurationError(
            f"artifact '{artifact}' does not support format '{as_}'",
            code="dataset.format_mismatch",
            context={"artifact": artifact, "format": as_},
        )


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


def resolve_training_set_filename(*, artifact: str, split: str, as_: str) -> str:
    if artifact in ROUTE_ARTIFACTS:
        return f"{split}.jsonl.gz"
    if as_ == "reaction_records":
        return f"{split}.jsonl.gz"
    return f"{split}.rsmi.txt.gz"


def resolve_training_set_root(
    *,
    dataset: TrainingDatasetName,
    release: str,
    cache_dir: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir) / release
    if cache_dir is not None:
        return Path(cache_dir) / dataset / release
    return DEFAULT_TRAINING_SET_CACHE_DIR / dataset / release


def resolve_hosted_data_root(*, cache_dir: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if cache_dir is not None:
        return Path(cache_dir)
    return DEFAULT_HOSTED_DATA_CACHE_DIR


def build_training_set_url(
    *,
    base_url: str,
    dataset: str,
    release: str | None = None,
    artifact: str | None = None,
    filename: str,
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


def download_sha256sums(url: str, destination: Path) -> dict[str, str]:
    download_url_to_path(url, destination)
    return load_sha256sums(destination)


def download_hosted_data_file(
    *,
    relative_path: Path,
    cache_dir: Path | None,
    output_dir: Path | None,
    base_url: str,
) -> Path:
    root_dir = resolve_hosted_data_root(cache_dir=cache_dir, output_dir=output_dir)
    local_path = root_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path = root_dir / "SHA256SUMS"
    checksums_url = build_hosted_data_url(base_url=base_url, relative_path=Path("SHA256SUMS"))

    expected_hashes = load_or_download_checksums(checksums_path=checksums_path, checksums_url=checksums_url)
    expected_hash = expected_hashes.get(str(relative_path))
    if expected_hash is None:
        expected_hashes = download_sha256sums(checksums_url, checksums_path)
        expected_hash = expected_hashes.get(str(relative_path))
    if expected_hash is None:
        raise DatasetResolutionError(
            f"hosted data file is not published: {relative_path}",
            code="dataset.file_not_published",
            context={"relative_path": str(relative_path), "checksums_url": checksums_url},
        )

    if local_path.exists() and sha256_file(local_path) == expected_hash:
        return local_path

    download_url = build_hosted_data_url(base_url=base_url, relative_path=relative_path)
    download_url_to_path(download_url, local_path)

    actual_hash = sha256_file(local_path)
    if actual_hash != expected_hash:
        invalidate_cached_download(local_path=local_path, checksums_path=checksums_path)
        raise DatasetVerificationError(
            f"downloaded hosted data file failed integrity verification: {relative_path.name}",
            code="dataset.hash_mismatch",
            context={
                "relative_path": str(relative_path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            },
        )
    return local_path


def load_or_download_checksums(*, checksums_path: Path, checksums_url: str) -> dict[str, str]:
    return (
        load_sha256sums(checksums_path)
        if checksums_path.exists()
        else download_sha256sums(checksums_url, checksums_path)
    )


def invalidate_cached_download(*, local_path: Path, checksums_path: Path) -> None:
    local_path.unlink(missing_ok=True)
    checksums_path.unlink(missing_ok=True)


def load_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetDownloadError(
            f"failed to read dataset checksums from {path}",
            code="dataset.checksums_read_failed",
            context={"path": str(path)},
        ) from exc

    output: dict[str, str] = {}
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
        output[filename.strip()] = sha256
    return output


def build_hosted_data_url(*, base_url: str, relative_path: Path) -> str:
    parts = [base_url.rstrip("/")]
    parts.extend(quote(part, safe=".") for part in relative_path.parts)
    return "/".join(parts)


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


def download_url_to_path(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with (
            urlopen(Request(url, headers={"User-Agent": DEFAULT_DATASET_USER_AGENT})) as response,
            tmp_path.open("wb") as handle,
        ):
            while chunk := response.read(8192):
                handle.write(chunk)
    except HTTPError as exc:
        raise DatasetResolutionError(
            f"failed to download hosted file from {url}: HTTP {exc.code}",
            code="dataset.file_http_error",
            context={"url": url, "status": exc.code},
        ) from exc
    except URLError as exc:
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


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


__all__ = [
    "DEFAULT_TRAINING_SET_BASE_URL",
    "DEFAULT_HOSTED_DATA_BASE_URL",
    "DEFAULT_DATASET_USER_AGENT",
    "DownloadedBenchmarkAssets",
    "DownloadedTrainingSet",
    "StockFormat",
    "TrainingArtifactName",
    "TrainingDatasetName",
    "TrainingReactionRecord",
    "TrainingRouteRecord",
    "TrainingSetFormat",
    "TrainingSplitName",
    "download_benchmark",
    "download_benchmark_assets",
    "download_stock",
    "download_training_set",
    "download_training_set_info",
    "load_training_set",
    "resolve_latest_training_set_release",
    "resolve_training_set_release",
]
