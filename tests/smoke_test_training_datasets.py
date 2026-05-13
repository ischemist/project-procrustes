"""
smoke tests for the public training-set guide.

these tests mirror the documented `retrocast.datasets` workflow so the guide
and the import surface stay aligned.
"""

from __future__ import annotations

import pytest

from retrocast.curation.training.records import TrainingReactionRecord, TrainingRouteRecord
from retrocast.datasets import (
    download_training_set,
    download_training_set_info,
    load_training_set,
    resolve_latest_training_set_release,
)
from retrocast.io import iter_training_reaction_records, iter_training_route_records
from tests.helpers_training_datasets import (
    write_latest_pointer,
    write_training_reaction_artifact,
    write_training_route_artifact,
)


@pytest.mark.integration
class TestTrainingSetGuideSmoke:
    def test_public_import_surface(self):
        from retrocast.datasets import TrainingReactionRecord as PublicTrainingReactionRecord
        from retrocast.datasets import TrainingRouteRecord as PublicTrainingRouteRecord

        assert PublicTrainingRouteRecord is TrainingRouteRecord
        assert PublicTrainingReactionRecord is TrainingReactionRecord

    def test_quick_start_route_example(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)

        train_routes = load_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="routes",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert len(train_routes) == 1
        assert train_routes[0].get_structural_signature() == route.get_structural_signature()

    def test_one_step_reaction_example(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        write_training_reaction_artifact(remote_root)

        train_reactions = load_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_records",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )

        assert len(train_reactions) == 1
        assert train_reactions[0].mapped_smiles == "c>o>cc"

    def test_streaming_and_metadata_examples(self, tmp_path, synthetic_route_factory):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)
        route = synthetic_route_factory("linear", depth=1)
        write_training_route_artifact(remote_root, route)
        write_training_reaction_artifact(remote_root)

        route_info = download_training_set_info(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="route_records",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        downloaded_path = download_training_set(
            "paroutes",
            artifact="reaction-holdout-n1-n5",
            split="training",
            as_="routes",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        route_records = list(iter_training_route_records(route_info.path))
        reaction_path = download_training_set(
            "paroutes",
            artifact="single-step-reaction-holdout-n1-n5",
            split="training",
            as_="reaction_records",
            base_url=remote_root.resolve().as_uri(),
            cache_dir=tmp_path / "cache",
        )
        reaction_records = list(iter_training_reaction_records(reaction_path))

        assert route_info.resolved_release == "v2026-05-12"
        assert route_info.path == downloaded_path
        assert len(route_records) == 1
        assert route_records[0].route_signature == route.get_structural_signature()
        assert len(reaction_records) == 1
        assert reaction_records[0].mapped_smiles == "c>o>cc"

    def test_release_resolution_example(self, tmp_path):
        remote_root = tmp_path / "remote"
        write_latest_pointer(remote_root)

        assert resolve_latest_training_set_release("paroutes", base_url=remote_root.resolve().as_uri()) == "v2026-05-12"
