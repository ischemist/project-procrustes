from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrocast.curation.training import (
    RawRouteSource,
    TrainingReactionRecord,
    TrainingReactionSource,
    TrainingRouteRecord,
)
from retrocast.datasets import (
    download_benchmark,
    download_benchmark_assets,
    download_stock,
    download_training_set,
    load_training_set,
)
from retrocast.exceptions import ConfigurationError, DatasetVerificationError
from retrocast.io import save_json_gz, save_jsonl_gz, save_lines_gz, save_stock_files
from retrocast.io.provenance import calculate_file_hash
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.typing import InchiKeyStr, ReactionSmilesStr, SmilesStr
from tests.helpers import _synthetic_inchikey


def write_training_route_artifact(remote_root: Path, route) -> None:
    artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "reaction-holdout-n1-n5"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    record = TrainingRouteRecord(
        id="paroutes-reaction-holdout-n1-n5-000001",
        split="training",
        route=route,
        sources=[
            RawRouteSource(
                dataset="all-routes",
                raw_index=0,
                raw_route_hash="route-hash-1",
                patent_id="patent-1",
            )
        ],
    )
    save_jsonl_gz([record.to_json_dict()], artifact_dir / "training.jsonl.gz")
    save_jsonl_gz([], artifact_dir / "validation.jsonl.gz")
    save_jsonl_gz([record.to_json_dict()], artifact_dir / "all.jsonl.gz")
    write_manifest_and_checksums(artifact_dir)


def write_training_reaction_artifact(remote_root: Path) -> None:
    artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "single-step-reaction-holdout-n1-n5"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    record = TrainingReactionRecord(
        id="paroutes-rxn-000001",
        split="training",
        reactants=[SmilesStr("c")],
        product=SmilesStr("cc"),
        mapped_smiles=ReactionSmilesStr("c>o>cc"),
        condition_slot="o",
        condition_slot_smiles=[SmilesStr("o")],
        sources=[TrainingReactionSource(route_id="paroutes-reaction-holdout-n1-n5-000001", step_index=0)],
    )
    save_jsonl_gz([record.model_dump(mode="json")], artifact_dir / "training.jsonl.gz")
    save_jsonl_gz([], artifact_dir / "validation.jsonl.gz")
    save_jsonl_gz([record.model_dump(mode="json")], artifact_dir / "all.jsonl.gz")
    save_lines_gz(["c>o>cc"], artifact_dir / "training.rsmi.txt.gz")
    save_lines_gz([], artifact_dir / "validation.rsmi.txt.gz")
    save_lines_gz(["c>o>cc"], artifact_dir / "all.rsmi.txt.gz")
    write_manifest_and_checksums(artifact_dir)


def write_manifest_and_checksums(artifact_dir: Path) -> None:
    output_files = {}
    for path in sorted(p for p in artifact_dir.iterdir() if p.is_file()):
        output_files[path.stem.replace(".", "_")] = FileInfo(
            label=path.stem, path=str(path), file_hash=calculate_file_hash(path)
        )

    manifest = Manifest(
        action="tests.write_training_artifact",
        release_name=artifact_dir.name,
        output_files=output_files,
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True))

    files = sorted(p for p in artifact_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    sha256sums = "".join(f"{calculate_file_hash(path)}  {path.name}\n" for path in files)
    (artifact_dir / "SHA256SUMS").write_text(sha256sums, encoding="utf-8")


def write_latest_pointer(remote_root: Path) -> None:
    dataset_root = remote_root / "paroutes"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "latest.json").write_text(
        json.dumps({"dataset": "paroutes", "latest_release": "v2026-05-12"}, indent=2),
        encoding="utf-8",
    )


def write_hosted_data_tree(remote_root: Path) -> None:
    definitions_dir = remote_root / "1-benchmarks" / "definitions"
    stocks_dir = remote_root / "1-benchmarks" / "stocks"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    stocks_dir.mkdir(parents=True, exist_ok=True)

    benchmark = BenchmarkSet(
        name="test-bench",
        stock_name="test-stock",
        targets={
            "t1": BenchmarkTarget(
                id="t1",
                smiles=SmilesStr("cc"),
                inchi_key=InchiKeyStr(_synthetic_inchikey("cc")),
                acceptable_routes=[],
            )
        },
    )
    save_json_gz(benchmark.model_dump(mode="json"), definitions_dir / "test-bench.json.gz")
    (definitions_dir / "test-bench.manifest.json").write_text("{}", encoding="utf-8")

    save_stock_files(
        stock={InchiKeyStr(_synthetic_inchikey("c")): SmilesStr("c")},
        stock_name="test-stock",
        output_dir=stocks_dir,
    )
    write_data_checksums(remote_root)


def write_data_checksums(remote_root: Path) -> None:
    files = sorted(path for path in remote_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    sha256sums = "".join(f"{calculate_file_hash(path)}  {path.relative_to(remote_root)}\n" for path in files)
    (remote_root / "SHA256SUMS").write_text(sha256sums, encoding="utf-8")


@pytest.mark.integration
class TestTrainingDatasets:
    def test_load_training_routes_from_latest_release(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        routes = load_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="routes",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert len(routes) == 1
        assert routes[0].get_structural_signature() == route.get_structural_signature()

    def test_load_training_route_records(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        records = load_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="route_records",
            release="v2026-05-12",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert len(records) == 1
        assert records[0].id == "paroutes-reaction-holdout-n1-n5-000001"
        assert records[0].route_signature == route.get_structural_signature()

    def test_download_reaction_smiles_and_redownload_after_cache_corruption(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        write_training_reaction_artifact(remote_root)

        path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_smiles",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        original_bytes = path.read_bytes()

        path.write_bytes(b"corrupted")
        restored_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_smiles",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert restored_path.read_bytes() == original_bytes

    def test_redownloads_checksums_after_remote_artifact_changes(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        write_training_reaction_artifact(remote_root)
        cache_dir = tmp_path / "cache"

        first_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_smiles",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )
        original_bytes = first_path.read_bytes()

        assert first_path.exists()
        artifact_dir = remote_root / "paroutes" / "v2026-05-12" / "single-step-reaction-holdout-n1-n5"
        save_lines_gz(["c>n>cc"], artifact_dir / "training.rsmi.txt.gz")
        write_manifest_and_checksums(artifact_dir)
        first_path.unlink()

        with pytest.raises(DatasetVerificationError):
            download_training_set(
                "paroutes",
                artifact="single-step-reaction-holdout-n1-n5",
                split="training",
                as_="reaction_smiles",
                release="latest",
                base_url=remote_root.resolve().as_uri(),
                cache_dir=cache_dir,
            )

        second_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_smiles",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )

        assert second_path.read_bytes() != original_bytes

    def test_download_training_set_to_explicit_output_dir(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        output_dir = tmp_path / "project-data"
        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="routes",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            output_dir=output_dir,
        )

        assert path == output_dir / "paroutes" / "v2026-05-12" / "reaction-holdout-n1-n5" / "training.jsonl.gz"
        assert path.exists()
        assert (path.parent / "SHA256SUMS").exists()

    def test_rejects_incompatible_artifact_format_combo(self, tmp_path):
        with pytest.raises(ConfigurationError, match="does not support format"):
            load_training_set(
                "paroutes",
                artifact="reaction-holdout-n1-n5",
                split="training",
                as_="reaction_smiles",
                release="v2026-05-12",
                base_url=(tmp_path / "remote").resolve().as_uri(),
                cache_dir=tmp_path / "cache",
            )

    def test_download_benchmark(self, tmp_path):
        remote_root = tmp_path / "remote-data"
        write_hosted_data_tree(remote_root)

        path = download_benchmark(
            "test-bench",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert path == tmp_path / "cache" / "1-benchmarks" / "definitions" / "test-bench.json.gz"
        assert path.exists()

    def test_download_stock(self, tmp_path):
        remote_root = tmp_path / "remote-data"
        write_hosted_data_tree(remote_root)

        path = download_stock(
            "test-stock",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert path == tmp_path / "cache" / "1-benchmarks" / "stocks" / "test-stock.csv.gz"
        assert path.exists()

    def test_redownloads_hosted_data_checksums_after_remote_file_changes(self, tmp_path):
        remote_root = tmp_path / "remote-data"
        write_hosted_data_tree(remote_root)
        cache_dir = tmp_path / "cache"

        first_path = download_stock(
            "test-stock",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )
        original_bytes = first_path.read_bytes()

        save_stock_files(
            stock={InchiKeyStr(_synthetic_inchikey("cc")): SmilesStr("cc")},
            stock_name="test-stock",
            output_dir=remote_root / "1-benchmarks" / "stocks",
        )
        write_data_checksums(remote_root)
        first_path.unlink()

        with pytest.raises(DatasetVerificationError):
            download_stock(
                "test-stock",
                base_url=remote_root.resolve().as_uri(),
                cache_dir=cache_dir,
            )

        second_path = download_stock(
            "test-stock",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )

        assert second_path.read_bytes() != original_bytes

    def test_download_benchmark_assets_downloads_declared_stock(self, tmp_path):
        remote_root = tmp_path / "remote-data"
        write_hosted_data_tree(remote_root)

        assets = download_benchmark_assets(
            "test-bench",
            base_url=remote_root.resolve().as_uri(),
            output_dir=tmp_path / "project-data",
        )

        assert assets.benchmark_path.exists()
        assert assets.stock_path == tmp_path / "project-data" / "1-benchmarks" / "stocks" / "test-stock.csv.gz"
        assert assets.stock_path is not None
        assert assets.stock_path.exists()
