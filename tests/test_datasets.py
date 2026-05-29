from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.datasets import (
    download_benchmark,
    download_benchmark_assets,
    download_stock,
    download_training_set,
    resolve_latest_training_set_release,
)
from retrocast.exceptions import ConfigurationError, DatasetVerificationError
from retrocast.io import save_benchmark
from retrocast.models import Benchmark, Target, TaskConstraints
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


def write_hosted_data(remote_root: Path) -> None:
    benchmark_path = remote_root / "1-benchmarks" / "definitions" / "small.json.gz"
    stock_path = remote_root / "1-benchmarks" / "stocks" / "test-stock.csv.gz"
    benchmark_path.parent.mkdir(parents=True)
    stock_path.parent.mkdir(parents=True)
    save_benchmark(benchmark(), benchmark_path)
    with gzip.open(stock_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "InChIKey"])
        writer.writerow(["C", get_inchi_key("C")])
    write_sha256sums(
        remote_root / "SHA256SUMS",
        root=remote_root,
        paths=[
            Path("1-benchmarks/definitions/small.json.gz"),
            Path("1-benchmarks/stocks/test-stock.csv.gz"),
        ],
    )


def benchmark() -> Benchmark:
    smiles = canonicalize_smiles("CCO")
    target = Target(id="ethanol", smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
    return Benchmark(
        name="small",
        targets={target.id: target},
        default_constraints=TaskConstraints(stock="test-stock"),
    )


def write_sha256sums(path: Path, *, root: Path, paths: list[Path]) -> None:
    lines = []
    for relative_path in paths:
        digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        lines.append(f"{digest} {relative_path.as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
