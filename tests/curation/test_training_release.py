from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.curation.training import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    RawRouteSource,
    TrainingReactionRecord,
    TrainingReactionReleaseBuilder,
    TrainingReactionSource,
    TrainingRouteRecord,
    TrainingRouteReleaseBuilder,
    TrainingSetBuildConfig,
    adapt_training_routes,
    write_training_reaction_release,
    write_training_release,
)
from retrocast.curation.training.audit import (
    RouteReleaseFiles,
    audit_route_release_sanity,
    audit_single_step_release_if_present,
)
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import load_training_route_records, save_jsonl_gz, save_lines_gz
from retrocast.models import Molecule, Reaction, Route
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr


def test_schema_v2_training_route_and_reaction_release_roundtrip(tmp_path: Path) -> None:
    adapted, stats = adapt_training_routes(
        [raw_route()], "all", id_width=2, collect_reactions=True, show_progress=False
    )
    config = TrainingSetBuildConfig(holdout_mode="route", val_fraction=0.0, show_progress=False)

    route_result = TrainingRouteReleaseBuilder(
        all_routes=adapted,
        all_adaptation=stats,
        holdout_routes={},
        holdout_adaptation={},
        config=config,
    ).build()
    write_training_release(
        result=route_result,
        output_dir=tmp_path,
        source_paths=[],
        source_root=tmp_path,
        config=config,
    )

    route_release_dir = tmp_path / "route-holdout-n1-n5"
    loaded_routes = load_training_route_records(route_release_dir / "all.jsonl.gz")
    assert len(loaded_routes) == 1
    assert loaded_routes[0].route_signature == route_result.records[0].route.signature()
    assert (route_release_dir / "manifest.json").exists()
    route_manifest = json.loads((route_release_dir / "manifest.json").read_text())
    assert route_manifest["output_files"]["all"]["content_hash"] is not None

    reaction_config = TrainingSetBuildConfig(holdout_mode="reaction", val_fraction=0.0, show_progress=False)
    reaction_result = TrainingReactionReleaseBuilder(route_records=loaded_routes, config=reaction_config).build()
    write_training_reaction_release(
        result=reaction_result,
        output_dir=tmp_path,
        source_paths=[route_release_dir / "all.jsonl.gz"],
        source_root=tmp_path,
        config=reaction_config,
    )

    reaction_release_dir = tmp_path / "single-step-reaction-holdout-n1-n5"
    assert len(reaction_result.records) == 2
    assert (reaction_release_dir / "training.rsmi.txt.gz").exists()
    assert (reaction_release_dir / "manifest.json").exists()
    reaction_manifest = json.loads((reaction_release_dir / "manifest.json").read_text())
    assert reaction_manifest["output_files"]["all"]["content_hash"] is not None


def test_route_release_collapses_mapped_variants_and_preserves_sources() -> None:
    route_a = one_step_route("C.C>>CC", patent_id="patent-a")
    route_b = one_step_route("[CH3:1].[CH3:2]>>[CH3:1][CH3:2]", patent_id="patent-b")

    result = TrainingRouteReleaseBuilder(
        all_routes=[adapted_route("a", route_a), adapted_route("b", route_b)],
        all_adaptation=adaptation_stats(raw=2, adapted=2),
        holdout_routes={},
        holdout_adaptation={},
        config=TrainingSetBuildConfig(holdout_mode="route", val_fraction=0.0, show_progress=False),
    ).build()

    assert len(result.records) == 1
    assert result.summary["postprocessing"]["chemical_duplicates_removed"] == 0
    assert result.summary["postprocessing"]["mapped_smiles_variants_collapsed"] == 1
    record = result.records[0]
    assert [source.patent_id for source in record.sources] == ["patent-a", "patent-b"]
    assert record.route.annotations["source_patent_ids"] == ["patent-a", "patent-b"]
    assert "patent_id" not in record.route.annotations
    assert record.route.target.product_of is not None
    assert record.route.target.product_of.annotations["alternative_mapped_smiles"] == [
        "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]"
    ]


def test_route_release_keeps_condition_variants_separate() -> None:
    route_a = one_step_route("C.C>O>CC", condition_slot_smiles=["O"])
    route_b = one_step_route("C.C>N>CC", condition_slot_smiles=["N"])

    result = TrainingRouteReleaseBuilder(
        all_routes=[adapted_route("a", route_a), adapted_route("b", route_b)],
        all_adaptation=adaptation_stats(raw=2, adapted=2),
        holdout_routes={},
        holdout_adaptation={},
        config=TrainingSetBuildConfig(holdout_mode="route", val_fraction=0.0, show_progress=False),
    ).build()

    assert len(result.records) == 2
    assert result.summary["postprocessing"]["mapped_smiles_variants_collapsed"] == 0


def test_route_audit_allows_condition_distinct_same_structure() -> None:
    records = [
        route_record("oxygen", one_step_route("C.C>O>CC", condition_slot_smiles=["O"]), split="training"),
        route_record("nitrogen", one_step_route("C.C>N>CC", condition_slot_smiles=["N"]), split="training"),
    ]

    audit_route_release_sanity(
        release_name="route-holdout-n1-n5",
        files=RouteReleaseFiles(all=records, training=records, validation=[]),
        holdout={},
    )


def test_route_audit_rejects_release_identity_failures() -> None:
    duplicate_id_a = route_record(
        "same-id-a", one_step_route("C.C>O>CC", condition_slot_smiles=["O"]), split="training"
    )
    duplicate_id_b = route_record(
        "same-id-b", one_step_route("C.C>N>CC", condition_slot_smiles=["N"]), split="training"
    )
    duplicate_id_b = duplicate_id_b.model_copy(update={"id": duplicate_id_a.id})
    duplicate_identity_a = route_record("same-route-a", one_step_route("C.C>>CC"), split="training")
    duplicate_identity_b = route_record("same-route-b", one_step_route("C.C>>CC"), split="training")
    duplicate_validation_a = route_record(
        "same-val-a", one_step_route("N.N>>NN", product="NN", reactant="N"), split="validation"
    )
    duplicate_validation_b = route_record(
        "same-val-b", one_step_route("N.N>>NN", product="NN", reactant="N"), split="validation"
    )
    wrong_split = route_record("wrong-split", one_step_route("N.N>>NN", product="NN", reactant="N"), split="validation")
    wrong_validation = route_record(
        "wrong-validation", one_step_route("O.O>>OO", product="OO", reactant="O"), split="training"
    )
    all_records = [
        duplicate_id_a,
        duplicate_id_b,
        duplicate_identity_a,
        duplicate_identity_b,
        duplicate_validation_a,
        duplicate_validation_b,
    ]

    with pytest.raises(TrainingReleaseError) as error:
        audit_route_release_sanity(
            release_name="route-holdout-n1-n5",
            files=RouteReleaseFiles(
                all=all_records,
                training=all_records[:4] + [wrong_split],
                validation=[duplicate_validation_a, duplicate_validation_b, wrong_validation],
            ),
            holdout={},
        )

    assert error.value.code == "workflow.route_release_audit_failed"
    failures = error.value.context["failures"]
    assert failures["duplicate_record_ids_in_all"] == 1
    assert failures["duplicate_record_ids_in_splits"] == 1
    assert failures["all_split_id_delta"] == 2
    assert failures["training_file_wrong_split_records"] == 1
    assert failures["validation_file_wrong_split_records"] == 1
    assert failures["duplicate_training_route_identities"] == 1
    assert failures["duplicate_validation_route_identities"] == 1


def test_route_release_excises_reaction_holdout_overlap() -> None:
    full_route = two_step_route()
    holdout_route = one_step_route("C.C>>CC")

    result = TrainingRouteReleaseBuilder(
        all_routes=[adapted_route("full", full_route)],
        all_adaptation=adaptation_stats(raw=1, adapted=1),
        holdout_routes={"n1": [adapted_route("holdout", holdout_route)]},
        holdout_adaptation={"n1": adaptation_stats(raw=1, adapted=1)},
        config=TrainingSetBuildConfig(holdout_mode="reaction", val_fraction=0.0, show_progress=False),
    ).build()

    assert len(result.records) == 1
    assert result.records[0].route.target.smiles == SmilesStr(canonicalize_smiles("CCC"))
    assert result.records[0].route.depth() == 1
    assert result.summary["postprocessing"]["reaction_overlap"] == {
        "unique_reference_reaction_signatures": 1,
        "routes_with_overlapping_reactions": 1,
        "fragments_kept_after_excision": 1,
        "routes_fully_removed_after_excision": 0,
    }


def test_reaction_release_deduplicates_and_removes_validation_overlap() -> None:
    training_route = one_step_route("C.C>O>CC", condition_slot_smiles=["O"])
    validation_overlap = one_step_route("[CH3:1].[CH3:2]>O>[CH3:1][CH3:2]", condition_slot_smiles=["O"])
    validation_unique = one_step_route("N.N>S>NN", product="NN", reactant="N", condition_slot_smiles=["S"])
    config = TrainingSetBuildConfig(holdout_mode="reaction", val_fraction=0.0, show_progress=False)

    result = TrainingReactionReleaseBuilder(
        route_records=[
            route_record("train", training_route, split="training"),
            route_record("val-overlap", validation_overlap, split="validation"),
            route_record("val-unique", validation_unique, split="validation"),
        ],
        config=config,
    ).build()

    assert [record.split for record in result.records] == ["training", "validation"]
    assert result.records[1].product == SmilesStr(canonicalize_smiles("NN"))
    assert result.summary["reaction_postprocessing"]["validation"]["overlap_removed_from_validation"] == 1
    assert (
        result.summary["reaction_postprocessing"]["cross_split_overlap_before_cleanup"]["shared_reaction_identities"]
        == 1
    )
    assert (
        result.summary["reaction_postprocessing"]["cross_split_overlap_after_cleanup"]["shared_reaction_identities"]
        == 0
    )


def test_reaction_release_merges_exact_duplicate_sources() -> None:
    route = one_step_route("C.C>>CC")
    config = TrainingSetBuildConfig(holdout_mode="reaction", val_fraction=0.0, show_progress=False)

    result = TrainingReactionReleaseBuilder(
        route_records=[
            route_record("first", route, split="training"),
            route_record("second", route, split="training"),
        ],
        config=config,
    ).build()

    assert len(result.records) == 1
    assert [source.route_id for source in result.records[0].sources] == ["route-first", "route-second"]
    assert result.summary["reaction_postprocessing"]["training"]["chemical_duplicates_removed"] == 1
    assert result.summary["reaction_postprocessing"]["training"]["mapped_smiles_variants_collapsed"] == 0


def test_reaction_release_requires_reaction_holdout_and_mapped_reactions() -> None:
    route = one_step_route(None)

    with pytest.raises(TrainingReleaseError, match="requires") as config_error:
        TrainingReactionReleaseBuilder(
            route_records=[route_record("route", route, split="training")],
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        ).build()
    assert config_error.value.code == "workflow.single_step_requires_reaction_holdout"

    with pytest.raises(TrainingReleaseError, match="mapped_reaction_smiles") as mapped_error:
        TrainingReactionReleaseBuilder(
            route_records=[route_record("route", route, split="training")],
            config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
        ).build()
    assert mapped_error.value.code == "workflow.single_step_missing_mapped_smiles"


def test_single_step_audit_rejects_release_identity_failures(tmp_path: Path) -> None:
    release_dir = tmp_path / "single-step-reaction-holdout-n1-n5"
    training = reaction_record("train", "route-train", split="training")
    validation = reaction_record("validation", "missing-route", split="validation")
    save_jsonl_gz([training, validation], release_dir / "all.jsonl.gz")
    save_jsonl_gz([training], release_dir / "training.jsonl.gz")
    save_jsonl_gz([validation], release_dir / "validation.jsonl.gz")
    save_lines_gz([training.to_rsmi_line()], release_dir / "all.rsmi.txt.gz")
    save_lines_gz([training.to_rsmi_line()], release_dir / "training.rsmi.txt.gz")
    save_lines_gz([validation.to_rsmi_line()], release_dir / "validation.rsmi.txt.gz")

    with pytest.raises(TrainingReleaseError) as error:
        audit_single_step_release_if_present(release_root=tmp_path, parent_route_ids={"route-train"})

    assert error.value.code == "workflow.single_step_release_audit_failed"
    failures = error.value.context["failures"]
    assert failures["rsmi_count_mismatches"] == {"all": 1, "training": 0, "validation": 0}
    assert failures["cross_split_reaction_identity_overlap"] == 1
    assert failures["missing_parent_route_sources"] == 1


def test_training_release_builders_are_single_use() -> None:
    route_builder = TrainingRouteReleaseBuilder(
        all_routes=[],
        all_adaptation=adaptation_stats(raw=0, adapted=0),
        holdout_routes={},
        holdout_adaptation={},
        config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
    )
    route_builder.build()
    with pytest.raises(RuntimeError, match="single-use"):
        route_builder.build()

    reaction_builder = TrainingReactionReleaseBuilder(
        route_records=[],
        config=TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False),
    )
    reaction_builder.build()
    with pytest.raises(RuntimeError, match="single-use"):
        reaction_builder.build()


def test_adaptation_rejects_ambiguous_paroutes_reaction_hashes() -> None:
    raw = raw_route()
    root_metadata = raw["children"][0]["metadata"]
    child_metadata = raw["children"][0]["children"][1]["children"][0]["metadata"]
    root_metadata["reaction_hash"] = "same-hash"
    child_metadata["reaction_hash"] = "same-hash"

    with pytest.raises(TrainingReleaseError, match="reaction_hash"):
        adapt_training_routes([raw], "all", id_width=2, collect_reactions=True, show_progress=False)


def test_adaptation_counts_invalid_raw_routes() -> None:
    adapted, stats = adapt_training_routes([[], {}, {"smiles": ""}, raw_route()], "all", 2, False, False)

    assert len(adapted) == 1
    assert stats.raw_routes == 4
    assert stats.adapted_routes == 1
    assert stats.skipped_routes == 3
    assert stats.skipped_without_error_code == 0
    assert stats.failures_by_code == {"adapter.schema_invalid": 3}


def raw_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "C.CC>>CCO"},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                    {
                        "type": "mol",
                        "smiles": "CC",
                        "children": [
                            {
                                "type": "reaction",
                                "smiles": "CC",
                                "metadata": {"ID": "US123;2", "rsmi": "C>>CC"},
                                "children": [{"type": "mol", "smiles": "C", "in_stock": True, "children": []}],
                            }
                        ],
                    },
                ],
            }
        ],
    }


def one_step_route(
    mapped_smiles: str | None,
    *,
    product: str = "CC",
    reactant: str = "C",
    patent_id: str = "patent",
    condition_slot_smiles: list[str] | None = None,
) -> Route:
    annotations = {"source_id": f"{patent_id};1"}
    if condition_slot_smiles is not None:
        annotations["condition_slot_smiles"] = [canonicalize_smiles(value) for value in condition_slot_smiles]
    return Route(
        target=molecule(
            product,
            product_of=Reaction(
                reactants=[molecule(reactant), molecule(reactant)],
                mapped_reaction_smiles=ReactionSmilesStr(mapped_smiles) if mapped_smiles is not None else None,
                annotations=annotations,
            ),
        ),
        annotations={"patent_id": patent_id},
    )


def two_step_route() -> Route:
    intermediate = molecule(
        "CC",
        product_of=Reaction(
            reactants=[molecule("C"), molecule("C")],
            mapped_reaction_smiles=ReactionSmilesStr("C.C>>CC"),
            annotations={"source_id": "patent;1"},
        ),
    )
    return Route(
        target=molecule(
            "CCC",
            product_of=Reaction(
                reactants=[intermediate, molecule("C")],
                mapped_reaction_smiles=ReactionSmilesStr("C.CC>>CCC"),
                annotations={"source_id": "patent;2"},
            ),
        ),
        annotations={"patent_id": "patent"},
    )


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)), product_of=product_of)


def adapted_route(name: str, route: Route) -> AdaptedTrainingRoute:
    return AdaptedTrainingRoute(
        route=route,
        structural_signature=route.signature(),
        reaction_signatures=route.reaction_signatures(),
        source=RawRouteSource(
            dataset="all", raw_index=0, raw_route_hash=f"hash-{name}", patent_id=route.annotations.get("patent_id")
        ),
    )


def route_record(name: str, route: Route, *, split: str) -> TrainingRouteRecord:
    return TrainingRouteRecord(
        id=f"route-{name}", split=split, route=route, sources=[adapted_route(name, route).source]
    )


def reaction_record(name: str, route_id: str, *, split: str) -> TrainingReactionRecord:
    return TrainingReactionRecord(
        id=f"reaction-{name}",
        split=split,
        reactants=[SmilesStr(canonicalize_smiles("C")), SmilesStr(canonicalize_smiles("C"))],
        product=SmilesStr(canonicalize_smiles("CC")),
        mapped_smiles=ReactionSmilesStr("C.C>>CC"),
        condition_slot_smiles=[SmilesStr(canonicalize_smiles("O"))],
        sources=[TrainingReactionSource(route_id=route_id, step_index=1, reaction_id="rc:r:/", source_id="source")],
    )


def adaptation_stats(*, raw: int, adapted: int) -> AdaptationStatistics:
    return AdaptationStatistics(
        raw_routes=raw,
        adapted_routes=adapted,
        skipped_routes=raw - adapted,
        skipped_without_error_code=0,
        failures_by_code={},
    )
