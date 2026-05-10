from __future__ import annotations

import json
from typing import cast

import pytest

from retrocast.curation.training_sets import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    RawRouteSource,
    TrainingSetBuildConfig,
    adapt_training_routes,
    assign_train_val_splits,
    build_training_manifest,
    build_training_records_from_adapted,
    summarize_records,
)
from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr


def make_leaf(name: str) -> Molecule:
    return Molecule(
        smiles=SmilesStr(name),
        inchikey=InchiKeyStr(f"INCHI-{name}"),
    )


def make_route(name: str, depth: int) -> Route:
    current = make_leaf(f"{name}-0")
    for idx in range(1, depth + 1):
        current = Molecule(
            smiles=SmilesStr(f"{name}-{idx}"),
            inchikey=InchiKeyStr(f"INCHI-{name}-{idx}"),
            synthesis_step=ReactionStep(reactants=[current]),
        )
    return Route(target=current, rank=1)


def make_adapted_route(name: str, route: Route) -> AdaptedTrainingRoute:
    return AdaptedTrainingRoute(
        route=route,
        route_signature=route.get_signature(),
        reaction_signatures=route.get_reaction_signatures(),
        source=RawRouteSource(dataset="all-routes", raw_index=0, raw_route_hash=f"hash-{name}"),
    )


@pytest.mark.unit
class TestTrainingSetSplits:
    def test_assign_train_val_splits_is_deterministic_and_stratified(self):
        routes = {f"route-{idx}": make_route(f"r{idx}", depth=2) for idx in range(20)}
        routes.update({f"long-{idx}": make_route(f"m{idx}", depth=3) for idx in range(20)})

        first = assign_train_val_splits(routes, val_fraction=0.1, seed=17)
        second = assign_train_val_splits(routes, val_fraction=0.1, seed=17)

        assert first == second
        assert sum(split == "val" for split in first.values()) == 4
        assert sum(first[f"route-{idx}"] == "val" for idx in range(20)) == 2
        assert sum(first[f"long-{idx}"] == "val" for idx in range(20)) == 2

    def test_assign_train_val_splits_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match="val_fraction"):
            assign_train_val_splits({}, val_fraction=0, seed=1)

    def test_build_from_adapted_routes_reuses_adapted_inputs(self):
        route = make_route("keep", depth=1)
        heldout = make_route("heldout", depth=1)

        result = build_training_records_from_adapted(
            all_routes=[make_adapted_route("keep", route), make_adapted_route("heldout", heldout)],
            all_adaptation=AdaptationStatistics(
                raw_routes=2,
                adapted_routes=2,
                skipped_routes=3,
                skipped_without_error_code=1,
                failures_by_code={"adapter.schema_invalid": 2},
            ),
            heldout_routes={"n1": [make_adapted_route("heldout", heldout)]},
            heldout_adaptation={
                "n1": AdaptationStatistics(
                    raw_routes=1,
                    adapted_routes=1,
                    skipped_routes=0,
                    skipped_without_error_code=0,
                    failures_by_code={},
                )
            },
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        )

        assert [record.route_signature for record in result.records] == [route.get_signature()]
        assert result.summary["input"]["all_routes"] == 2
        assert result.summary["adaptation"]["all_routes"]["skipped_routes"] == 3
        assert result.summary["postprocessing"]["exact_route_matches_removed"] == 1
        assert result.summary["adaptation"]["all_routes"]["failures_by_code"] == {"adapter.schema_invalid": 2}

    def test_adapt_training_routes_tracks_failures_by_code(self):
        valid_route = {
            "type": "mol",
            "smiles": "CCO",
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": "CCO",
                    "metadata": {"ID": "US20150051201A1;outer", "rsmi": "CC.O>>CCO"},
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CC",
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": "CC",
                                    "metadata": {"ID": "US20150051201A1;inner", "rsmi": "C.C>>CC"},
                                    "children": [
                                        {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                                        {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                                    ],
                                }
                            ],
                        },
                        {"type": "mol", "smiles": "O", "in_stock": True, "children": []},
                    ],
                }
            ],
        }
        mixed_patent_route = json.loads(json.dumps(valid_route))
        mixed_patent_route["children"][0]["children"][0]["children"][0]["metadata"]["ID"] = (
            "SOME-OTHER-PATENT;1234;56789"
        )

        adapted_routes, stats = adapt_training_routes(
            [valid_route, {}, mixed_patent_route],
            dataset="all",
            id_width=6,
            collect_reactions=False,
            show_progress=False,
        )

        assert len(adapted_routes) == 1
        assert stats.raw_routes == 3
        assert stats.adapted_routes == 1
        assert stats.skipped_routes == 2
        assert stats.skipped_without_error_code == 0
        assert stats.failures_by_code == {
            "adapter.multiple_patents": 1,
            "adapter.schema_invalid": 1,
        }

    def test_reaction_holdout_excises_overlapping_reactions(self):
        leaf_a = make_leaf("a")
        mol_b = Molecule(
            smiles=SmilesStr("b"),
            inchikey=InchiKeyStr("INCHI-b"),
            synthesis_step=ReactionStep(reactants=[leaf_a]),
        )
        mol_c = Molecule(
            smiles=SmilesStr("c"),
            inchikey=InchiKeyStr("INCHI-c"),
            synthesis_step=ReactionStep(reactants=[mol_b]),
        )
        mol_d = Molecule(
            smiles=SmilesStr("d"),
            inchikey=InchiKeyStr("INCHI-d"),
            synthesis_step=ReactionStep(reactants=[mol_c]),
        )

        full_route = Route(target=mol_d, rank=1)
        heldout_route = Route(
            target=Molecule(
                smiles=SmilesStr("c"),
                inchikey=InchiKeyStr("INCHI-c"),
                synthesis_step=ReactionStep(
                    reactants=[
                        Molecule(
                            smiles=SmilesStr("b"),
                            inchikey=InchiKeyStr("INCHI-b"),
                        )
                    ]
                ),
            ),
            rank=1,
        )
        result = build_training_records_from_adapted(
            all_routes=[make_adapted_route("full", full_route)],
            all_adaptation=AdaptationStatistics(
                raw_routes=1,
                adapted_routes=1,
                skipped_routes=0,
                skipped_without_error_code=0,
                failures_by_code={},
            ),
            heldout_routes={"n1": [make_adapted_route("heldout", heldout_route)]},
            heldout_adaptation={
                "n1": AdaptationStatistics(
                    raw_routes=1,
                    adapted_routes=1,
                    skipped_routes=0,
                    skipped_without_error_code=0,
                    failures_by_code={},
                )
            },
            config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
        )

        record_signatures = {record.route_signature for record in result.records}
        record_targets = {record.route.target.inchikey for record in result.records}
        assert len(result.records) == 2
        assert result.summary["postprocessing"]["exact_route_matches_removed"] == 0
        assert result.summary["postprocessing"]["reaction_overlap"] == {
            "unique_reference_reaction_signatures": 1,
            "routes_with_overlapping_reactions": 1,
            "fragments_kept_after_excision": 2,
            "routes_fully_removed_after_excision": 0,
        }
        assert heldout_route.get_signature() not in record_signatures
        assert full_route.get_signature() not in record_signatures
        assert record_targets == {"INCHI-d", "INCHI-b"}


@pytest.mark.unit
class TestTrainingSetSummary:
    def test_summarize_records_counts_splits(self):
        class Record:
            def __init__(self, split: str) -> None:
                self.split = split

        summary = summarize_records(
            cast(
                list,
                [
                    Record("train"),
                    Record("train"),
                    Record("val"),
                ],
            )
        )

        assert summary["all_records"] == {
            "total": 3,
            "training_split": 2,
            "validation_split": 1,
        }

    def test_build_training_manifest_preserves_release_summary_shape(self, tmp_path):
        source_file = tmp_path / "input.json.gz"
        source_file.write_text("source", encoding="utf-8")
        output_file = tmp_path / "release" / "train.jsonl.gz"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("output", encoding="utf-8")

        manifest = build_training_manifest(
            release_name="route-heldout-n1-n5",
            files={"train": output_file},
            source_paths=[source_file],
            source_root=tmp_path,
            config=TrainingSetBuildConfig(
                holdout_mode="route",
                val_fraction=0.05,
                seed=17,
                show_progress=False,
            ),
            summary={"output": {"all_records": {"total": 1}}},
        )

        assert manifest["release_name"] == "route-heldout-n1-n5"
        assert manifest["summary"]["output"]["all_records"]["total"] == 1
        assert manifest["source_files"][0]["sha256"]
        assert manifest["output_files"]["train"]["path"] == "release/train.jsonl.gz"
        assert manifest["output_files"]["train"]["sha256"]
