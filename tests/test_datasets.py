from __future__ import annotations

import pytest

import retrocast.datasets as datasets_module
from retrocast.curation.training import (
    TrainingReactionRecord,
    TrainingRouteRecord,
)
from retrocast.datasets import (
    TrainingReactionRecord as PublicTrainingReactionRecord,
)
from retrocast.datasets import (
    TrainingRouteRecord as PublicTrainingRouteRecord,
)
from retrocast.datasets import (
    download_benchmark,
    download_benchmark_assets,
    download_stock,
    download_training_set,
    resolve_latest_training_set_release,
)
from retrocast.exceptions import ConfigurationError, DatasetVerificationError
from retrocast.io import (
    iter_training_reaction_smiles,
    iter_training_route_records,
    load_training_routes,
    save_lines_gz,
    save_stock_files,
)
from retrocast.typing import InchiKeyStr, SmilesStr
from tests.helpers import _synthetic_inchikey
from tests.helpers_training_datasets import (
    write_data_checksums,
    write_hosted_data_tree,
    write_latest_pointer,
    write_manifest_and_checksums,
    write_training_reaction_artifact,
    write_training_route_artifact,
)


@pytest.mark.integration
class TestTrainingDatasets:
    def test_show_download_progress_honors_explicit_flag(self):
        assert datasets_module.should_show_download_progress(True) is True
        assert datasets_module.should_show_download_progress(False) is False

    def test_show_download_progress_auto_detects_tty(self, monkeypatch):
        class FakeStderr:
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(datasets_module.sys, "stderr", FakeStderr())

        assert datasets_module.should_show_download_progress(None) is True

    def test_show_download_progress_auto_disables_without_tty(self, monkeypatch):
        class FakeStderr:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr(datasets_module.sys, "stderr", FakeStderr())

        assert datasets_module.should_show_download_progress(None) is False

    def test_public_dataset_module_reexports_training_record_models(self):
        assert PublicTrainingRouteRecord is TrainingRouteRecord
        assert PublicTrainingReactionRecord is TrainingReactionRecord

    def test_load_training_routes_from_latest_release(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        routes = load_training_routes(path)

        assert len(routes) == 1
        assert routes[0].get_structural_signature() == route.get_structural_signature()

    def test_load_training_route_records(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            release="v2026-05-12",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        records = list(iter_training_route_records(path))

        assert len(records) == 1
        assert records[0].id == "paroutes-reaction-holdout-n1-n5-000001"
        assert records[0].route_signature == route.get_structural_signature()

    def test_download_training_set_downloads_manifest_and_checksums(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert path == tmp_path / "cache" / "paroutes" / "v2026-05-12" / "reaction-holdout-n1-n5" / "training.jsonl.gz"
        assert (path.parent / "manifest.json").exists()
        assert (path.parent.parent / "SHA256SUMS").exists()

    def test_download_training_set_restores_missing_manifest(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        manifest_path = path.parent / "manifest.json"
        manifest_path.unlink()

        restored_path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert restored_path == path
        assert manifest_path.exists()

    def test_download_training_set_path_can_be_streamed_with_io_loader(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        records = list(iter_training_route_records(path))

        assert len(records) == 1
        assert records[0].route_signature == route.get_structural_signature()

    def test_download_reaction_smiles_and_redownload_after_cache_corruption(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        write_training_reaction_artifact(remote_root)

        path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            format="rsmi",
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
            format="rsmi",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert restored_path.read_bytes() == original_bytes

    def test_verified_download_path_can_be_streamed_with_io_loader(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        write_training_reaction_artifact(remote_root)
        cache_dir = tmp_path / "cache"

        path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            format="rsmi",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )
        first_lines = list(iter_training_reaction_smiles(path))

        original_bytes = path.read_bytes()
        path.write_bytes(b"corrupted")

        restored_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            format="rsmi",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )
        second_lines = list(iter_training_reaction_smiles(restored_path))

        assert first_lines == ["c>o>cc"]
        assert second_lines == first_lines
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
            format="rsmi",
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
                format="rsmi",
                release="latest",
                base_url=remote_root.resolve().as_uri(),
                cache_dir=cache_dir,
            )

        second_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            format="rsmi",
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

        output_dir = tmp_path / "project-data" / "paroutes"
        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            output_dir=output_dir,
        )

        assert path == output_dir / "v2026-05-12" / "reaction-holdout-n1-n5" / "training.jsonl.gz"
        assert path.exists()
        assert (path.parents[1] / "SHA256SUMS").exists()

    def test_download_training_set_to_explicit_cache_dir_keeps_shared_layout(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        cache_dir = tmp_path / "custom-cache"
        path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            release="latest",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=cache_dir,
        )

        assert path == cache_dir / "paroutes" / "v2026-05-12" / "reaction-holdout-n1-n5" / "training.jsonl.gz"

    def test_resolve_latest_training_set_release(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)

        assert resolve_latest_training_set_release("paroutes", base_url=remote_root.resolve().as_uri()) == "v2026-05-12"

    def test_rejects_incompatible_artifact_format_combo(self, tmp_path):
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
