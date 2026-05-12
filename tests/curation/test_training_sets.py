from __future__ import annotations

import json
from typing import cast

import pytest

from retrocast.adapters.paroutes_adapter import ConditionSlotParseStatistics
from retrocast.curation.training import (
    TRAINING_RELEASE_ACTION,
    AdaptationStatistics,
    AdaptedTrainingRoute,
    RawRouteSource,
    TrainingRouteRecord,
    TrainingRouteReleaseBuilder,
    TrainingSetBuildConfig,
    adapt_training_routes,
    assign_train_val_splits,
    build_route_release_split_audit,
    build_training_manifest,
    build_training_reaction_records_from_route_records,
    render_route_release_split_audit_markdown,
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


def make_convergent_route(name: str) -> Route:
    left_leaf = make_leaf(f"{name}-left-leaf")
    right_leaf = make_leaf(f"{name}-right-leaf")
    left_intermediate = Molecule(
        smiles=SmilesStr(f"{name}-left"),
        inchikey=InchiKeyStr(f"INCHI-{name}-left"),
        synthesis_step=ReactionStep(reactants=[left_leaf]),
    )
    right_intermediate = Molecule(
        smiles=SmilesStr(f"{name}-right"),
        inchikey=InchiKeyStr(f"INCHI-{name}-right"),
        synthesis_step=ReactionStep(reactants=[right_leaf]),
    )
    target = Molecule(
        smiles=SmilesStr(f"{name}-target"),
        inchikey=InchiKeyStr(f"INCHI-{name}-target"),
        synthesis_step=ReactionStep(reactants=[left_intermediate, right_intermediate]),
    )
    return Route(target=target, rank=1)


def make_reaction_route(
    name: str,
    *,
    mapped_smiles: str,
    source_id: str = "rxn-1",
    condition_slot_smiles: list[str] | None = None,
    patent_id: str = "patent-default",
) -> Route:
    reactant = make_leaf(f"{name}-leaf")
    metadata = {"source_id": source_id}
    if condition_slot_smiles is not None:
        metadata["condition_slot_smiles"] = condition_slot_smiles

    target = Molecule(
        smiles=SmilesStr(f"{name}-target"),
        inchikey=InchiKeyStr(f"INCHI-{name}-target"),
        synthesis_step=ReactionStep(
            reactants=[reactant],
            mapped_smiles=mapped_smiles,
            metadata=metadata,
        ),
    )
    return Route(target=target, rank=1, metadata={"patent_id": patent_id})


def make_adapted_route(
    name: str,
    route: Route,
    *,
    patent_id: str | None = None,
) -> AdaptedTrainingRoute:
    patent_id = patent_id or f"patent-{name}"
    return AdaptedTrainingRoute(
        route=route,
        structural_signature=route.get_structural_signature(),
        reaction_signatures=route.get_reaction_signatures(),
        source=RawRouteSource(
            dataset="all-routes",
            raw_index=0,
            raw_route_hash=f"hash-{name}",
            patent_id=patent_id,
        ),
    )


def make_route_record(name: str, route: Route, *, split: str, patent_id: str | None = None) -> TrainingRouteRecord:
    return TrainingRouteRecord(
        id=f"route-{name}",
        split=split,
        route_signature=route.get_structural_signature(),
        content_hash=route.get_content_hash(),
        route=route,
        sources=[
            RawRouteSource(
                dataset="all-routes",
                raw_index=0,
                raw_route_hash=f"hash-{name}",
                patent_id=patent_id or f"patent-{name}",
            )
        ],
    )


@pytest.mark.unit
class TestTrainingSetSplits:
    def test_assign_train_val_splits_is_deterministic_and_stratified(self):
        routes = {f"route-{idx}": make_route(f"r{idx}", depth=2) for idx in range(20)}
        routes.update({f"long-{idx}": make_route(f"m{idx}", depth=3) for idx in range(20)})

        first = assign_train_val_splits(routes, val_fraction=0.1, seed=17)
        second = assign_train_val_splits(routes, val_fraction=0.1, seed=17)

        assert first == second
        assert sum(split == "validation" for split in first.values()) == 4
        assert sum(first[f"route-{idx}"] == "validation" for idx in range(20)) == 2
        assert sum(first[f"long-{idx}"] == "validation" for idx in range(20)) == 2

    def test_assign_train_val_splits_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match="val_fraction"):
            assign_train_val_splits({}, val_fraction=0, seed=1)

    def test_build_route_release_split_audit_and_render_markdown(self):
        linear_training = make_route_record("linear-training", make_route("linear", depth=2), split="training")
        linear_validation = make_route_record(
            "linear-validation", make_route("linear-val", depth=2), split="validation"
        )
        convergent_training = make_route_record(
            "convergent-training",
            make_convergent_route("conv"),
            split="training",
        )

        audit = build_route_release_split_audit(
            release_name="reaction-heldout-n1-n5",
            route_records=[linear_training, linear_validation, convergent_training],
        )

        assert audit.total_counts.training == 2
        assert audit.total_counts.validation == 1
        assert [(row.length, row.counts.training, row.counts.validation) for row in audit.by_length] == [(2, 2, 1)]
        assert [
            (row.has_convergent_reaction, row.counts.training, row.counts.validation) for row in audit.by_convergence
        ] == [
            (False, 1, 1),
            (True, 1, 0),
        ]
        assert [
            (row.length, row.has_convergent_reaction, row.counts.training, row.counts.validation)
            for row in audit.by_length_and_convergence
        ] == [
            (2, False, 1, 1),
            (2, True, 1, 0),
        ]

        markdown = render_route_release_split_audit_markdown(release_root_name="v2026-05-11", audits=[audit])
        assert "# release audit" in markdown
        assert "## `reaction-heldout-n1-n5`" in markdown
        assert "| `false` | 1 | 1 | 2 | 50.0000% | 66.6667% |" in markdown
        assert (
            "| length | non-conv train | non-conv val | non-conv all | non-conv val% | conv train | conv val | conv all | conv val% |"
            in markdown
        )
        assert "| 2 | 1 | 1 | 2 | 50.0000% | 1 | 0 | 1 | 0.0000% |" in markdown

    def test_build_from_adapted_routes_reuses_adapted_inputs(self):
        route = make_route("keep", depth=1)
        heldout = make_route("heldout", depth=1)

        result = TrainingRouteReleaseBuilder(
            all_routes=[make_adapted_route("keep", route), make_adapted_route("heldout", heldout)],
            all_adaptation=AdaptationStatistics(
                raw_routes=2,
                adapted_routes=2,
                skipped_routes=3,
                skipped_without_error_code=1,
                failures_by_code={"adapter.schema_invalid": 2},
                non_fatal_condition_slot_parse=ConditionSlotParseStatistics(
                    malformed_rsmi_count=4,
                    uncanonicalizable_token_count=7,
                    uncanonicalizable_tokens={"noise": 1},
                ),
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
        ).build()

        assert [record.route_signature for record in result.records] == [route.get_structural_signature()]
        assert result.summary["input"]["all_routes"] == 2
        assert result.summary["adaptation"]["all_routes"]["skipped_routes"] == 3
        assert result.summary["postprocessing"]["exact_route_matches_removed"] == 1
        assert result.summary["adaptation"]["all_routes"]["failures_by_code"] == {"adapter.schema_invalid": 2}
        assert result.summary["adaptation"]["all_routes"]["non_fatal_condition_slot_parse"] == {
            "malformed_rsmi_count": 4,
            "uncanonicalizable_token_count": 7,
            "distinct_uncanonicalizable_token_count": 1,
        }

    def test_build_from_adapted_routes_merges_patent_only_duplicates(self):
        route_a = make_reaction_route(
            "dup",
            mapped_smiles="C.C>O>CC",
            patent_id="patent-a",
        )
        route_b = make_reaction_route(
            "dup",
            mapped_smiles="C.C>O>CC",
            patent_id="patent-b",
        )

        result = TrainingRouteReleaseBuilder(
            all_routes=[
                make_adapted_route("dup-a", route_a, patent_id="patent-a"),
                make_adapted_route("dup-b", route_b, patent_id="patent-b"),
            ],
            all_adaptation=AdaptationStatistics(
                raw_routes=2,
                adapted_routes=2,
                skipped_routes=0,
                skipped_without_error_code=0,
                failures_by_code={},
            ),
            heldout_routes={},
            heldout_adaptation={},
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        ).build()

        assert len(result.records) == 1
        assert result.summary["postprocessing"]["chemical_duplicates_removed"] == 1
        assert result.summary["postprocessing"]["mapped_smiles_variants_collapsed"] == 0
        assert "patent_id" not in result.records[0].route.metadata
        assert result.records[0].route.metadata["source_patent_ids"] == ["patent-a", "patent-b"]
        assert [source.patent_id for source in result.records[0].sources] == ["patent-a", "patent-b"]

        record_json = result.records[0].to_json_dict()
        assert record_json["source"]["patent_ids"] == ["patent-a", "patent-b"]

    def test_build_from_adapted_routes_keeps_condition_variants_separate(self):
        route_a = make_reaction_route(
            "cond",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
        )
        route_b = make_reaction_route(
            "cond",
            mapped_smiles="C.C>N>CC",
            condition_slot_smiles=["N"],
        )

        result = TrainingRouteReleaseBuilder(
            all_routes=[
                make_adapted_route("cond-a", route_a),
                make_adapted_route("cond-b", route_b),
            ],
            all_adaptation=AdaptationStatistics(
                raw_routes=2,
                adapted_routes=2,
                skipped_routes=0,
                skipped_without_error_code=0,
                failures_by_code={},
            ),
            heldout_routes={},
            heldout_adaptation={},
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        ).build()

        assert len(result.records) == 2
        assert len({record.split for record in result.records}) == 1
        assert result.summary["postprocessing"]["chemical_duplicates_removed"] == 0
        assert result.summary["postprocessing"]["mapped_smiles_variants_collapsed"] == 0

    def test_build_from_adapted_routes_collapses_mapped_smiles_variants(self):
        canonical_route = make_reaction_route(
            "mapped",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
            patent_id="patent-a",
        )
        duplicate_canonical_route = make_reaction_route(
            "mapped",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
            patent_id="patent-b",
        )
        variant_route = make_reaction_route(
            "mapped",
            mapped_smiles="[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]",
            condition_slot_smiles=["O"],
            patent_id="patent-c",
        )

        result = TrainingRouteReleaseBuilder(
            all_routes=[
                make_adapted_route("mapped-a", canonical_route, patent_id="patent-a"),
                make_adapted_route(
                    "mapped-b",
                    duplicate_canonical_route,
                    patent_id="patent-b",
                ),
                make_adapted_route("mapped-c", variant_route, patent_id="patent-c"),
            ],
            all_adaptation=AdaptationStatistics(
                raw_routes=3,
                adapted_routes=3,
                skipped_routes=0,
                skipped_without_error_code=0,
                failures_by_code={},
            ),
            heldout_routes={},
            heldout_adaptation={},
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        ).build()

        assert len(result.records) == 1
        assert result.summary["postprocessing"]["chemical_duplicates_removed"] == 1
        assert result.summary["postprocessing"]["mapped_smiles_variants_collapsed"] == 1

        step = result.records[0].route.target.synthesis_step
        assert step is not None
        assert result.records[0].route.metadata["source_patent_ids"] == ["patent-a", "patent-b", "patent-c"]
        assert step.mapped_smiles == "C.C>O>CC"
        assert step.metadata["alternative_mapped_smiles"] == ["[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]"]

    def test_build_training_reaction_records_flattens_and_deduplicates(self):
        canonical_route = make_reaction_route(
            "single-step",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
            patent_id="patent-a",
        )
        variant_route = make_reaction_route(
            "single-step",
            mapped_smiles="[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]",
            condition_slot_smiles=["O"],
            patent_id="patent-b",
        )

        result = build_training_reaction_records_from_route_records(
            [
                make_route_record("single-step-a", canonical_route, split="training", patent_id="patent-a"),
                make_route_record("single-step-b", variant_route, split="training", patent_id="patent-b"),
            ],
            config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
        )

        assert result.release_name == "single-step-reaction-heldout-n1-n5"
        assert len(result.records) == 1
        assert result.summary["reaction_postprocessing"]["training"]["flattened_reactions"] == 2
        assert result.summary["reaction_postprocessing"]["training"]["chemical_duplicates_removed"] == 0
        assert result.summary["reaction_postprocessing"]["training"]["mapped_smiles_variants_collapsed"] == 1

        record = result.records[0]
        assert record.reactants == ["single-step-leaf"]
        assert record.product == "single-step-target"
        assert record.mapped_smiles == "C.C>O>CC"
        assert record.alternative_mapped_smiles == ["[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]"]
        assert record.condition_slot_smiles == ["O"]
        assert len(record.sources) == 2
        assert [source.patent_ids for source in record.sources] == [["patent-a"], ["patent-b"]]

    def test_build_training_reaction_records_keeps_condition_variants_separate(self):
        route_a = make_reaction_route(
            "single-step-cond",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
        )
        route_b = make_reaction_route(
            "single-step-cond",
            mapped_smiles="C.C>N>CC",
            condition_slot_smiles=["N"],
        )

        result = build_training_reaction_records_from_route_records(
            [
                make_route_record("single-step-cond-a", route_a, split="training"),
                make_route_record("single-step-cond-b", route_b, split="training"),
            ],
            config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
        )

        assert len(result.records) == 2
        assert result.summary["reaction_postprocessing"]["training"]["mapped_smiles_variants_collapsed"] == 0

    def test_build_training_reaction_records_requires_reaction_holdout_config(self):
        route = make_reaction_route(
            "single-step",
            mapped_smiles="C.C>O>CC",
        )

        with pytest.raises(ValueError, match="holdout_mode='reaction'"):
            build_training_reaction_records_from_route_records(
                [make_route_record("single-step", route, split="training")],
                config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
            )

    def test_build_training_reaction_records_from_route_records_removes_validation_overlap(self):
        training_route = make_reaction_route(
            "shared",
            mapped_smiles="C.C>O>CC",
            condition_slot_smiles=["O"],
            patent_id="patent-training",
        )
        validation_route = make_reaction_route(
            "shared",
            mapped_smiles="[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]",
            condition_slot_smiles=["O"],
            patent_id="patent-validation",
        )
        validation_unique_route = make_reaction_route(
            "unique",
            mapped_smiles="N.N>S>NN",
            condition_slot_smiles=["S"],
            patent_id="patent-unique",
        )

        result = build_training_reaction_records_from_route_records(
            [
                make_route_record("training", training_route, split="training", patent_id="patent-training"),
                make_route_record("validation", validation_route, split="validation", patent_id="patent-validation"),
                make_route_record(
                    "validation-unique", validation_unique_route, split="validation", patent_id="patent-unique"
                ),
            ],
            config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
        )

        assert [record.split for record in result.records] == ["training", "validation"]
        assert result.records[1].product == validation_unique_route.target.smiles
        assert result.summary["reaction_postprocessing"]["training"]["flattened_reactions"] == 1
        assert result.summary["reaction_postprocessing"]["validation"]["flattened_reactions"] == 2
        assert result.summary["reaction_postprocessing"]["validation"]["overlap_removed_from_validation"] == 1
        assert result.summary["reaction_postprocessing"]["cross_split_overlap_before_cleanup"] == {
            "shared_exact_reaction_signatures": 0,
            "shared_reaction_identities": 1,
            "training_records_with_shared_identity": 1,
            "validation_records_with_shared_identity": 1,
        }
        assert result.summary["reaction_postprocessing"]["cross_split_overlap_after_cleanup"] == {
            "shared_exact_reaction_signatures": 0,
            "shared_reaction_identities": 0,
            "training_records_with_shared_identity": 0,
            "validation_records_with_shared_identity": 0,
        }
        assert result.summary["output"]["all_records"] == {
            "total": 2,
            "training": 1,
            "validation": 1,
        }

    def test_adapt_training_routes_tracks_failures_by_code(self):
        valid_route = {
            "type": "mol",
            "smiles": "CCO",
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": "CCO",
                    "metadata": {
                        "ID": "US20150051201A1;outer",
                        "rsmi": "CC.O>>CCO",
                        "reaction_hash": "outer-hash",
                    },
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CC",
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": "CC",
                                    "metadata": {
                                        "ID": "US20150051201A1;inner",
                                        "rsmi": "C.C>>CC",
                                        "reaction_hash": "inner-hash",
                                    },
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
        assert stats.non_fatal_condition_slot_parse is not None
        assert stats.non_fatal_condition_slot_parse.malformed_rsmi_count == 0
        assert stats.non_fatal_condition_slot_parse.uncanonicalizable_token_count == 0
        assert stats.non_fatal_condition_slot_parse.distinct_uncanonicalizable_token_count == 0
        assert adapted_routes[0].source.patent_id == "US20150051201A1"

    def test_adapt_training_routes_aggregates_condition_slot_parse_noise(self, caplog):
        noisy_route = {
            "type": "mol",
            "smiles": "CCO",
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": "CCO",
                    "metadata": {
                        "ID": "US20150051201A1;outer",
                        "rsmi": "CC.O>CC(C)CC1=C(CC(C)C)[AlH3]1>CCO",
                        "reaction_hash": "outer-hash",
                    },
                    "children": [
                        {"type": "mol", "smiles": "CC", "in_stock": True, "children": []},
                        {"type": "mol", "smiles": "O", "in_stock": True, "children": []},
                    ],
                }
            ],
        }

        with caplog.at_level("INFO"):
            adapted_routes, stats = adapt_training_routes(
                [noisy_route],
                dataset="all",
                id_width=6,
                collect_reactions=False,
                show_progress=False,
            )

        assert len(adapted_routes) == 1
        assert stats.adapted_routes == 1
        assert stats.non_fatal_condition_slot_parse is not None
        assert stats.non_fatal_condition_slot_parse.malformed_rsmi_count == 0
        assert stats.non_fatal_condition_slot_parse.uncanonicalizable_token_count == 1
        assert stats.non_fatal_condition_slot_parse.distinct_uncanonicalizable_token_count == 1
        assert "could not canonicalize condition-slot token" not in caplog.text
        assert (
            "PaRoutes condition-slot parsing for all skipped 0 malformed rsmi slots and 1 uncanonicalizable tokens"
        ) in caplog.text

        manifest_dict = stats.to_manifest_dict()
        assert manifest_dict["non_fatal_condition_slot_parse"] == {
            "malformed_rsmi_count": 0,
            "uncanonicalizable_token_count": 1,
            "distinct_uncanonicalizable_token_count": 1,
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
        result = TrainingRouteReleaseBuilder(
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
        ).build()

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
        assert heldout_route.get_structural_signature() not in record_signatures
        assert full_route.get_structural_signature() not in record_signatures
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
                    Record("training"),
                    Record("training"),
                    Record("validation"),
                ],
            )
        )

        assert summary["all_records"] == {
            "total": 3,
            "training": 2,
            "validation": 1,
        }

    def test_build_training_manifest_preserves_release_summary_shape(self, tmp_path):
        source_file = tmp_path / "input.json.gz"
        source_file.write_text("source", encoding="utf-8")
        output_file = tmp_path / "release" / "training.jsonl.gz"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("output", encoding="utf-8")

        manifest = build_training_manifest(
            release_name="route-heldout-n1-n5",
            files={"training": output_file},
            source_paths=[source_file],
            source_root=tmp_path,
            config=TrainingSetBuildConfig(
                holdout_mode="route",
                val_fraction=0.05,
                seed=17,
                show_progress=False,
            ),
            summary={"output": {"all_records": {"total": 1}}},
            action=TRAINING_RELEASE_ACTION,
        )

        assert manifest["release_name"] == "route-heldout-n1-n5"
        assert manifest["summary"]["output"]["all_records"]["total"] == 1
        assert manifest["source_files"][0]["sha256"]
        assert manifest["output_files"]["training"]["path"] == "release/training.jsonl.gz"
        assert manifest["output_files"]["training"]["sha256"]
