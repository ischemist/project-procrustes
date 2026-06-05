from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.datasets import (
    build_hosted_data_url,
    build_training_set_url,
    download_benchmark,
    download_benchmark_assets,
    download_hosted_data_target,
    download_stock,
    download_training_data,
    download_training_set,
    load_sha256sums,
    published_training_files,
    resolve_expected_hash,
    resolve_hosted_data_root,
    resolve_latest_training_set_release,
    resolve_training_set_filename,
    resolve_training_set_root,
    should_show_download_progress,
    validate_training_dataset_request,
    write_response_with_progress,
)
from retrocast.exceptions import (
    ArtifactFormatError,
    ConfigurationError,
    DatasetResolutionError,
    DatasetVerificationError,
)
from retrocast.io import save_benchmark
from retrocast.models import Benchmark, StockTerminationConstraint, Target
from retrocast.typing import InChIKeyStr, SmilesStr


def test_download_training_set_resolves_latest_and_verifies_files(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    write_training_release(remote_root)

    path = download_training_set(
        "paroutes",
        artifact="single-step-reaction-holdout-n1-n5",
        split="training",
        format="rsmi",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )

    assert path == (
        tmp_path / "cache" / "paroutes" / "v2026-05-12" / "single-step-reaction-holdout-n1-n5" / "training.rsmi.txt.gz"
    )
    assert path.exists()
    assert (path.parent / "manifest.json").exists()
    assert resolve_latest_training_set_release("paroutes", base_url=remote_root.resolve().as_uri()) == "v2026-05-12"


def test_download_training_set_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not support format"):
        download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            format="rsmi",
            release="v2026-05-12",
            base_url=(tmp_path / "remote").resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )


def test_download_training_set_allows_n1_all_only_artifacts(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    write_n1_test_release(remote_root)

    path = download_training_set(
        "paroutes",
        artifact="n1-single-step-reactions",
        split="all",
        format="rsmi",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )

    assert path == tmp_path / "cache" / "paroutes" / "v2026-05-12" / "n1-single-step-reactions" / "all.rsmi.txt.gz"
    assert path.exists()
    with pytest.raises(ConfigurationError) as exc_info:
        validate_training_dataset_request(dataset="paroutes", artifact="n1-routes", split="training", format="jsonl")
    assert exc_info.value.code == "dataset.split_mismatch"


def test_download_training_data_expands_artifact_from_release_checksums(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    write_mixed_training_release(remote_root)

    paths = download_training_data(
        "paroutes",
        artifact="single-step-route-holdout-n1-n5",
        format="jsonl",
        omit=("validation",),
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )

    assert paths == [
        tmp_path / "cache" / "paroutes" / "v2026-05-12" / "single-step-route-holdout-n1-n5" / "all.jsonl.gz",
        tmp_path / "cache" / "paroutes" / "v2026-05-12" / "single-step-route-holdout-n1-n5" / "training.jsonl.gz",
        tmp_path / "cache" / "paroutes" / "v2026-05-12" / "single-step-route-holdout-n1-n5" / "manifest.json",
    ]
    assert all(path.exists() for path in paths)


def test_published_training_files_include_manifest_for_artifact_defaults(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    write_mixed_training_release(remote_root)

    files = published_training_files(
        checksums_path=tmp_path / "cache" / "SHA256SUMS",
        checksums_url=(remote_root / "paroutes" / "v2026-05-12" / "SHA256SUMS").resolve().as_uri(),
        artifact="single-step-route-holdout-n1-n5",
        split=None,
        format=None,
        omit=(),
    )

    assert [file.key for file in files] == [
        "single-step-route-holdout-n1-n5/all.jsonl.gz",
        "single-step-route-holdout-n1-n5/training.jsonl.gz",
        "single-step-route-holdout-n1-n5/validation.jsonl.gz",
        "single-step-route-holdout-n1-n5/all.rsmi.txt.gz",
        "single-step-route-holdout-n1-n5/manifest.json",
    ]


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {"dataset": "bad", "artifact": "reaction-holdout-n1-n5", "split": "training", "format": "jsonl"},
            "dataset.unsupported_dataset",
        ),
        (
            {"dataset": "paroutes", "artifact": "bad", "split": "training", "format": "jsonl"},
            "dataset.unsupported_artifact",
        ),
        (
            {"dataset": "paroutes", "artifact": "reaction-holdout-n1-n5", "split": "bad", "format": "jsonl"},
            "dataset.unsupported_split",
        ),
        (
            {"dataset": "paroutes", "artifact": "reaction-holdout-n1-n5", "split": "training", "format": "bad"},
            "dataset.unsupported_format",
        ),
    ],
)
def test_training_dataset_request_reports_specific_invalid_field(kwargs: dict[str, str], code: str) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        validate_training_dataset_request(**kwargs)

    assert exc_info.value.code == code


def test_download_training_set_redownloads_after_cached_corruption(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    write_training_release(remote_root)
    cache_dir = tmp_path / "cache"

    path = download_training_set(
        "paroutes",
        artifact="single-step-reaction-holdout-n1-n5",
        split="training",
        format="rsmi",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=cache_dir,
    )
    original_bytes = path.read_bytes()
    path.write_bytes(b"corrupted")

    restored = download_training_set(
        "paroutes",
        artifact="single-step-reaction-holdout-n1-n5",
        split="training",
        format="rsmi",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=cache_dir,
    )

    assert restored.read_bytes() == original_bytes


def test_download_benchmark_stock_and_assets(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-data"
    write_hosted_data(remote_root)

    benchmark_path = download_benchmark(
        "small",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )
    stock_path = download_stock(
        "test-stock",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )
    assets = download_benchmark_assets(
        "small",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache-assets",
    )

    assert benchmark_path.exists()
    assert stock_path.exists()
    assert assets.benchmark_path.exists()
    assert assets.stock_path is not None
    assert assets.stock_path.exists()


def test_download_hosted_data_target_expands_benchmark_dependencies(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-data"
    write_hosted_data(remote_root, benchmark_name="mkt-lin-500", stock_name="buyables-stock")

    paths = download_hosted_data_target(
        "mkt-lin-500",
        base_url=remote_root.resolve().as_uri(),
        cache_dir=tmp_path / "cache",
    )

    assert paths == [
        tmp_path / "cache" / "1-benchmarks" / "definitions" / "mkt-lin-500.json.gz",
        tmp_path / "cache" / "1-benchmarks" / "stocks" / "buyables-stock.csv.gz",
    ]
    assert all(path.exists() for path in paths)


def test_download_benchmark_and_stock_include_manifests_when_requested(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-data"
    write_hosted_data(remote_root, include_manifests=True)

    download_benchmark(
        "small", base_url=remote_root.resolve().as_uri(), cache_dir=tmp_path / "cache", include_manifest=True
    )
    download_stock(
        "test-stock", base_url=remote_root.resolve().as_uri(), cache_dir=tmp_path / "cache", include_manifest=True
    )

    assert (tmp_path / "cache" / "1-benchmarks" / "definitions" / "small.manifest.json").exists()
    assert (tmp_path / "cache" / "1-benchmarks" / "stocks" / "test-stock.manifest.json").exists()


def test_download_benchmark_assets_allows_benchmark_without_single_stock(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-data"
    write_hosted_data(remote_root, benchmark_value=benchmark_without_stock())

    assets = download_benchmark_assets("small", base_url=remote_root.resolve().as_uri(), cache_dir=tmp_path / "cache")

    assert assets.benchmark_path.exists()
    assert assets.stock_path is None


def test_download_detects_hash_mismatch(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-data"
    write_hosted_data(remote_root)
    stock_path = remote_root / "1-benchmarks" / "stocks" / "test-stock.csv.gz"
    stock_path.write_bytes(b"changed after checksum")

    with pytest.raises(DatasetVerificationError):
        download_stock(
            "test-stock",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )


def test_latest_release_rejects_invalid_metadata_payloads(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    latest_path = remote_root / "paroutes" / "latest.json"
    latest_path.parent.mkdir(parents=True)

    latest_path.write_text(json.dumps({"dataset": "other", "latest_release": "v1"}), encoding="utf-8")
    with pytest.raises(DatasetResolutionError) as mismatch:
        resolve_latest_training_set_release("paroutes", base_url=remote_root.resolve().as_uri())
    assert mismatch.value.code == "dataset.latest_dataset_mismatch"

    latest_path.write_text(json.dumps({"dataset": "paroutes"}), encoding="utf-8")
    with pytest.raises(ArtifactFormatError) as invalid:
        resolve_latest_training_set_release("paroutes", base_url=remote_root.resolve().as_uri())
    assert invalid.value.code == "dataset.invalid_latest_payload"


def test_checksum_resolution_refreshes_stale_cache_and_reports_missing_key(tmp_path: Path) -> None:
    checksums_path = tmp_path / "cache" / "SHA256SUMS"
    remote_checksums = tmp_path / "remote" / "SHA256SUMS"
    checksums_path.parent.mkdir()
    remote_checksums.parent.mkdir()
    checksums_path.write_text("0" * 64 + " stale.txt\n", encoding="utf-8")
    remote_checksums.write_text("1" * 64 + " wanted.txt\n", encoding="utf-8")

    resolved = resolve_expected_hash(
        checksums_path=checksums_path,
        checksums_url=remote_checksums.resolve().as_uri(),
        checksum_key="wanted.txt",
        missing_message="missing wanted",
        missing_context={"name": "wanted.txt"},
    )
    assert resolved == "1" * 64

    with pytest.raises(DatasetResolutionError) as exc_info:
        resolve_expected_hash(
            checksums_path=checksums_path,
            checksums_url=remote_checksums.resolve().as_uri(),
            checksum_key="absent.txt",
            missing_message="missing absent",
            missing_context={"name": "absent.txt"},
        )
    assert exc_info.value.code == "dataset.file_not_published"


def test_dataset_url_root_filename_and_checksum_helpers_are_explicit(tmp_path: Path) -> None:
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text("\nabc file.txt\n", encoding="utf-8")

    assert (
        resolve_training_set_filename(artifact="single-step-reaction-holdout-n1-n5", split="validation", format="rsmi")
        == "validation.rsmi.txt.gz"
    )
    assert (
        resolve_training_set_root(dataset="paroutes", release="v1", cache_dir=tmp_path / "cache", output_dir=None)
        == tmp_path / "cache" / "paroutes" / "v1"
    )
    assert (
        resolve_training_set_root(dataset="paroutes", release="v1", cache_dir=None, output_dir=tmp_path / "out")
        == tmp_path / "out" / "v1"
    )
    assert resolve_hosted_data_root(cache_dir=tmp_path / "cache", output_dir=None) == tmp_path / "cache"
    assert resolve_hosted_data_root(cache_dir=None, output_dir=tmp_path / "out") == tmp_path / "out"
    assert (
        build_training_set_url(
            base_url="https://example.test/root/",
            dataset="pa routes",
            release="v 1",
            artifact="art",
            filename="x.json.gz",
        )
        == "https://example.test/root/pa%20routes/v%201/art/x.json.gz"
    )
    assert (
        build_hosted_data_url(base_url="https://example.test/root/", relative_path=Path("a b/file.json.gz"))
        == "https://example.test/root/a%20b/file.json.gz"
    )
    assert load_sha256sums(checksums) == {"file.txt": "abc"}

    checksums.write_text("not-enough-fields\n", encoding="utf-8")
    with pytest.raises(ArtifactFormatError):
        load_sha256sums(checksums)


def test_default_dataset_cache_roots_use_runtime_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache-a"))
    assert (
        resolve_training_set_root(dataset="paroutes", release="v1", cache_dir=None, output_dir=None)
        == tmp_path / "cache-a" / "training-sets" / "paroutes" / "v1"
    )
    assert resolve_hosted_data_root(cache_dir=None, output_dir=None) == tmp_path / "cache-a" / "data"

    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache-b"))
    assert (
        resolve_training_set_root(dataset="paroutes", release="v1", cache_dir=None, output_dir=None)
        == tmp_path / "cache-b" / "training-sets" / "paroutes" / "v1"
    )
    assert resolve_hosted_data_root(cache_dir=None, output_dir=None) == tmp_path / "cache-b" / "data"


def test_progress_visibility_and_writer_handle_non_tty_and_unknown_lengths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenStderr:
        def isatty(self) -> bool:
            raise OSError("no tty")

    class Response:
        headers = {"Content-Length": "not-an-int"}

        def __init__(self) -> None:
            self._chunks = [b"abc", b""]

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    monkeypatch.setattr("retrocast.datasets.sys.stderr", BrokenStderr())
    assert should_show_download_progress(None) is False
    monkeypatch.undo()

    destination = tmp_path / "download.bin"
    with destination.open("wb") as handle:
        write_response_with_progress(response=Response(), handle=handle, description="download")

    assert should_show_download_progress(False) is False
    assert should_show_download_progress(True) is True
    assert destination.read_bytes() == b"abc"


def write_training_release(remote_root: Path) -> None:
    artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "single-step-reaction-holdout-n1-n5"
    artifact_dir.mkdir(parents=True)
    (remote_root / "paroutes").mkdir(exist_ok=True)
    (remote_root / "paroutes" / "latest.json").write_text(
        json.dumps({"dataset": "paroutes", "latest_release": "v2026-05-12"}),
        encoding="utf-8",
    )
    with gzip.open(artifact_dir / "training.rsmi.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write("c>o>cc\n")
    (artifact_dir / "manifest.json").write_text('{"schema_version":"2"}', encoding="utf-8")
    write_sha256sums(
        remote_root / "paroutes" / "v2026-05-12" / "SHA256SUMS",
        root=remote_root / "paroutes" / "v2026-05-12",
        paths=[
            Path("single-step-reaction-holdout-n1-n5/training.rsmi.txt.gz"),
            Path("single-step-reaction-holdout-n1-n5/manifest.json"),
        ],
    )


def write_n1_test_release(remote_root: Path) -> None:
    artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "n1-single-step-reactions"
    artifact_dir.mkdir(parents=True)
    (remote_root / "paroutes").mkdir(exist_ok=True)
    (remote_root / "paroutes" / "latest.json").write_text(
        json.dumps({"dataset": "paroutes", "latest_release": "v2026-05-12"}),
        encoding="utf-8",
    )
    with gzip.open(artifact_dir / "all.rsmi.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write("c>o>cc\n")
    (artifact_dir / "manifest.json").write_text('{"schema_version":"2"}', encoding="utf-8")
    write_sha256sums(
        remote_root / "paroutes" / "v2026-05-12" / "SHA256SUMS",
        root=remote_root / "paroutes" / "v2026-05-12",
        paths=[
            Path("n1-single-step-reactions/all.rsmi.txt.gz"),
            Path("n1-single-step-reactions/manifest.json"),
        ],
    )


def write_mixed_training_release(remote_root: Path) -> None:
    artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "single-step-route-holdout-n1-n5"
    artifact_dir.mkdir(parents=True)
    (remote_root / "paroutes").mkdir(exist_ok=True)
    (remote_root / "paroutes" / "latest.json").write_text(
        json.dumps({"dataset": "paroutes", "latest_release": "v2026-05-12"}),
        encoding="utf-8",
    )
    paths = [
        Path("single-step-route-holdout-n1-n5/all.jsonl.gz"),
        Path("single-step-route-holdout-n1-n5/training.jsonl.gz"),
        Path("single-step-route-holdout-n1-n5/validation.jsonl.gz"),
        Path("single-step-route-holdout-n1-n5/all.rsmi.txt.gz"),
    ]
    for path in paths:
        with gzip.open(remote_root / "paroutes" / "v2026-05-12" / path, "wt", encoding="utf-8") as handle:
            handle.write("{}\n")
    (artifact_dir / "manifest.json").write_text('{"schema_version":"2"}', encoding="utf-8")
    write_sha256sums(
        remote_root / "paroutes" / "v2026-05-12" / "SHA256SUMS",
        root=remote_root / "paroutes" / "v2026-05-12",
        paths=[*paths, Path("single-step-route-holdout-n1-n5/manifest.json")],
    )


def write_hosted_data(
    remote_root: Path,
    *,
    include_manifests: bool = False,
    benchmark_value: Benchmark | None = None,
    benchmark_name: str = "small",
    stock_name: str = "test-stock",
) -> None:
    benchmark_path = remote_root / "1-benchmarks" / "definitions" / f"{benchmark_name}.json.gz"
    stock_path = remote_root / "1-benchmarks" / "stocks" / f"{stock_name}.csv.gz"
    benchmark_path.parent.mkdir(parents=True)
    stock_path.parent.mkdir(parents=True)
    save_benchmark(benchmark_value or benchmark(), benchmark_path)
    with gzip.open(stock_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "InChIKey"])
        writer.writerow(["C", get_inchi_key("C")])
    paths = [
        Path(f"1-benchmarks/definitions/{benchmark_name}.json.gz"),
        Path(f"1-benchmarks/stocks/{stock_name}.csv.gz"),
    ]
    if include_manifests:
        benchmark_manifest = remote_root / "1-benchmarks" / "definitions" / f"{benchmark_name}.manifest.json"
        stock_manifest = remote_root / "1-benchmarks" / "stocks" / f"{stock_name}.manifest.json"
        benchmark_manifest.write_text('{"artifact":"benchmark"}', encoding="utf-8")
        stock_manifest.write_text('{"artifact":"stock"}', encoding="utf-8")
        paths.extend(
            [
                Path(f"1-benchmarks/definitions/{benchmark_name}.manifest.json"),
                Path(f"1-benchmarks/stocks/{stock_name}.manifest.json"),
            ]
        )
    write_sha256sums(
        remote_root / "SHA256SUMS",
        root=remote_root,
        paths=paths,
    )


def benchmark() -> Benchmark:
    smiles = canonicalize_smiles("CCO")
    target = Target(id="ethanol", smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
    return Benchmark(
        name="small",
        targets={target.id: target},
        default_constraints=[StockTerminationConstraint(stock="test-stock")],
    )


def benchmark_without_stock() -> Benchmark:
    smiles = canonicalize_smiles("CCO")
    target = Target(id="ethanol", smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
    return Benchmark(name="small", targets={target.id: target})


def write_sha256sums(path: Path, *, root: Path, paths: list[Path]) -> None:
    lines = []
    for relative_path in paths:
        digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        lines.append(f"{digest} {relative_path.as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
