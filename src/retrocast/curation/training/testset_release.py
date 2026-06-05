from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from retrocast.curation.training.records import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    TestReactionRecord,
    TestRouteRecord,
    TestSetName,
    TrainingReactionSource,
)
from retrocast.exceptions import TrainingReleaseError
from retrocast.hashing import hash_json
from retrocast.io import ContentType, create_manifest, save_jsonl_gz, save_lines_gz
from retrocast.typing import ReactionSmilesStr, SmilesStr

TEST_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-test-set-release"


def build_test_route_records(
    *, dataset: TestSetName, routes: Sequence[AdaptedTrainingRoute], route_prefix: str = "paroutes"
) -> list[TestRouteRecord]:
    width = max(5, len(str(len(routes))))
    return [
        TestRouteRecord(
            id=f"{route_prefix}-{dataset}-routes-{index:0{width}d}",
            dataset=dataset,
            route=route.route,
            sources=[route.source],
        )
        for index, route in enumerate(routes, start=1)
    ]


def build_test_reaction_records(
    *, dataset: TestSetName, route_records: Sequence[TestRouteRecord], route_prefix: str = "paroutes"
) -> list[TestReactionRecord]:
    records: list[TestReactionRecord] = []
    for route_record in route_records:
        if route_record.dataset != dataset:
            raise TrainingReleaseError(
                "test single-step release route dataset does not match requested dataset",
                code="workflow.test_single_step_dataset_mismatch",
                context={"dataset": dataset, "route_id": route_record.id, "route_dataset": route_record.dataset},
            )
        for step_index, reaction in enumerate(route_record.route.iter_reactions(), start=1):
            mapped_smiles = reaction.value.mapped_reaction_smiles
            if mapped_smiles is None:
                raise TrainingReleaseError(
                    f"test single-step release requires mapped_reaction_smiles; missing on route {route_record.id} reaction {reaction.id()}",
                    code="workflow.test_single_step_missing_mapped_smiles",
                    context={"route_id": route_record.id, "reaction_id": reaction.id()},
                )
            annotations = reaction.value.annotations
            condition_slot_smiles = _string_list_annotation(
                annotations,
                "condition_slot_smiles",
                route_id=route_record.id,
                reaction_id=reaction.id(),
            )
            alternative_mapped_smiles = _string_list_annotation(
                annotations,
                "alternative_mapped_smiles",
                route_id=route_record.id,
                reaction_id=reaction.id(),
            )
            records.append(
                TestReactionRecord(
                    id=f"{route_prefix}-{dataset}-single-step-reactions-{len(records) + 1:06d}",
                    dataset=dataset,
                    reactants=[reactant.value.smiles for reactant in reaction.reactants()],
                    product=reaction.product().value.smiles,
                    mapped_smiles=mapped_smiles,
                    condition_slot=annotations.get("condition_slot")
                    if isinstance(annotations.get("condition_slot"), str)
                    else None,
                    condition_slot_smiles=[SmilesStr(value) for value in condition_slot_smiles],
                    sources=[
                        TrainingReactionSource(
                            route_id=route_record.id,
                            step_index=step_index,
                            reaction_id=reaction.id(),
                            source_id=annotations.get("source_id")
                            if isinstance(annotations.get("source_id"), str)
                            else None,
                        )
                    ],
                    alternative_mapped_smiles=[
                        ReactionSmilesStr(value)
                        for value in alternative_mapped_smiles
                        if isinstance(value, str) and value != mapped_smiles
                    ],
                )
            )
    return records


def write_test_route_release(
    *,
    dataset: TestSetName,
    records: Sequence[TestRouteRecord],
    adaptation: AdaptationStatistics,
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
) -> None:
    release_name = f"{dataset}-routes"
    release_dir = output_dir / release_name
    all_path = release_dir / "all.jsonl.gz"
    manifest_path = release_dir / "manifest.json"
    save_jsonl_gz(records, all_path)
    depths = Counter(record.route.depth() for record in records)
    summary = {
        "adaptation": adaptation.to_manifest_dict(),
        "output": {
            "all_records": {"total": len(records)},
            "by_depth": {str(depth): depths[depth] for depth in sorted(depths)},
        },
    }
    manifest = create_manifest(
        action=TEST_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            (
                "all",
                all_path,
                records,
                ContentType.UNKNOWN,
                hash_json(
                    sorted(record.route.content_signature(fields=("mapped_reaction_smiles",)) for record in records)
                ),
            )
        ],
        root_dir=source_root,
        parameters={"dataset": dataset, "artifact": release_name},
        statistics=summary["output"],
        summary=summary,
        release_name=release_name,
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def write_test_reaction_release(
    *,
    dataset: TestSetName,
    records: Sequence[TestReactionRecord],
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
) -> None:
    release_name = f"{dataset}-single-step-reactions"
    release_dir = output_dir / release_name
    all_path = release_dir / "all.jsonl.gz"
    all_rsmi_path = release_dir / "all.rsmi.txt.gz"
    manifest_path = release_dir / "manifest.json"
    save_jsonl_gz(records, all_path)
    save_lines_gz((record.to_rsmi_line() for record in records), all_rsmi_path)
    summary = {"output": {"all_records": {"total": len(records)}}}
    manifest = create_manifest(
        action=TEST_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            (
                "all",
                all_path,
                records,
                ContentType.UNKNOWN,
                hash_json(sorted((str(record.mapped_smiles), _condition_identity(record)) for record in records)),
            ),
            (
                "all_rsmi",
                all_rsmi_path,
                [record.to_rsmi_line() for record in records],
                ContentType.UNKNOWN,
                hash_json(sorted(record.to_rsmi_line() for record in records)),
            ),
        ],
        root_dir=source_root,
        parameters={"dataset": dataset, "artifact": release_name},
        statistics=summary["output"],
        summary=summary,
        release_name=release_name,
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def _string_list_annotation(annotations: dict[str, object], key: str, *, route_id: str, reaction_id: str) -> list[str]:
    value = annotations.get(key, [])
    if not isinstance(value, list):
        raise TrainingReleaseError(
            f"test single-step release annotation '{key}' must be a list",
            code=f"workflow.test_single_step_invalid_{key}",
            context={"route_id": route_id, "reaction_id": reaction_id},
        )
    return [item for item in value if isinstance(item, str)]


def _condition_identity(record: TestReactionRecord) -> tuple[str, ...]:
    condition_smiles = tuple(str(value) for value in record.condition_slot_smiles)
    if condition_smiles:
        return condition_smiles
    return (record.condition_slot,) if record.condition_slot is not None else ()
