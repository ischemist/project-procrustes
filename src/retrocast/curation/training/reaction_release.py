from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import (
    TrainingReactionBuildResult,
    TrainingReactionRecord,
    TrainingReactionSource,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
)
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import ContentType, create_manifest, save_jsonl_gz, save_lines_gz
from retrocast.typing import ReactionSmilesStr, SmilesStr

TRAINING_REACTION_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-single-step-release"


class TrainingReactionReleaseBuilder:
    def __init__(self, *, route_records: Sequence[TrainingRouteRecord], config: TrainingSetBuildConfig) -> None:
        self.route_records = route_records
        self.config = config
        self._started = False

    def build(self) -> TrainingReactionBuildResult:
        if self.config.holdout_mode != "reaction":
            raise TrainingReleaseError(
                "single-step training release requires TrainingSetBuildConfig(holdout_mode='reaction')",
                code="workflow.single_step_requires_reaction_holdout",
                context={"holdout_mode": self.config.holdout_mode},
            )
        if self._started:
            raise RuntimeError("TrainingReactionReleaseBuilder instances are single-use")
        self._started = True

        training, training_postprocessing = self._build_split("training")
        validation_before, validation_postprocessing = self._build_split("validation")
        overlap_before = _summarize_cross_split_overlap(training, validation_before)
        training_keys = {_reaction_identity(record) for record in training}
        validation = [record for record in validation_before if _reaction_identity(record) not in training_keys]
        overlap_removed = len(validation_before) - len(validation)
        overlap_after = _summarize_cross_split_overlap(training, validation)
        if overlap_after["shared_reaction_identities"] or overlap_after["shared_exact_reaction_signatures"]:
            raise TrainingReleaseError(
                "single-step release validation split still overlaps with training after cleanup",
                code="workflow.single_step_validation_overlap",
                context=overlap_after,
            )
        records = _renumber([*training, *validation], prefix=self.config.route_prefix)
        return TrainingReactionBuildResult(
            release_name="single-step-reaction-holdout-n1-n5",
            records=records,
            summary={
                "reaction_postprocessing": {
                    "training": training_postprocessing,
                    "validation": {
                        **validation_postprocessing,
                        "overlap_removed_from_validation": overlap_removed,
                    },
                    "cross_split_overlap_before_cleanup": overlap_before,
                    "cross_split_overlap_after_cleanup": overlap_after,
                },
                "output": summarize_reaction_records(records),
            },
        )

    def _build_split(self, split: str) -> tuple[list[TrainingReactionRecord], dict[str, int]]:
        flattened = self._flatten(split)
        exact_unique, exact_duplicates_removed = _merge_exact_reaction_duplicates(flattened)
        transform_unique, mapped_variants_collapsed = _merge_transform_equivalent_reactions(exact_unique)
        return transform_unique, {
            "input_routes": sum(1 for record in self.route_records if record.split == split),
            "flattened_reactions": len(flattened),
            "chemical_duplicates_removed": exact_duplicates_removed,
            "mapped_smiles_variants_collapsed": mapped_variants_collapsed,
            "duplicate_reactions_removed": exact_duplicates_removed + mapped_variants_collapsed,
        }

    def _flatten(self, split: str) -> list[TrainingReactionRecord]:
        records = []
        for route_record in self.route_records:
            if route_record.split != split:
                continue
            for step_index, reaction in enumerate(route_record.route.iter_reactions(), start=1):
                mapped_smiles = reaction.value.mapped_reaction_smiles
                if mapped_smiles is None:
                    raise TrainingReleaseError(
                        f"single-step release requires mapped_reaction_smiles; missing on route {route_record.id} reaction {reaction.id()}",
                        code="workflow.single_step_missing_mapped_smiles",
                        context={"route_id": route_record.id, "reaction_id": reaction.id()},
                    )
                annotations = reaction.value.annotations
                records.append(
                    TrainingReactionRecord(
                        id="pending",
                        split=route_record.split,
                        reactants=[reactant.value.smiles for reactant in reaction.reactants()],
                        product=reaction.product().value.smiles,
                        mapped_smiles=mapped_smiles,
                        condition_slot=annotations.get("condition_slot")
                        if isinstance(annotations.get("condition_slot"), str)
                        else None,
                        condition_slot_smiles=[
                            SmilesStr(value)
                            for value in annotations.get("condition_slot_smiles", [])
                            if isinstance(value, str)
                        ],
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
                            for value in annotations.get("alternative_mapped_smiles", [])
                            if isinstance(value, str) and value != mapped_smiles
                        ],
                    )
                )
        return records


def write_training_reaction_release(
    *,
    result: TrainingReactionBuildResult,
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
    config: TrainingSetBuildConfig,
) -> None:
    release_dir = output_dir / result.release_name
    all_path = release_dir / "all.jsonl.gz"
    training_path = release_dir / "training.jsonl.gz"
    validation_path = release_dir / "validation.jsonl.gz"
    all_rsmi_path = release_dir / "all.rsmi.txt.gz"
    training_rsmi_path = release_dir / "training.rsmi.txt.gz"
    validation_rsmi_path = release_dir / "validation.rsmi.txt.gz"
    manifest_path = release_dir / "manifest.json"
    training = [record for record in result.records if record.split == "training"]
    validation = [record for record in result.records if record.split == "validation"]
    save_jsonl_gz(result.records, all_path)
    save_jsonl_gz(training, training_path)
    save_jsonl_gz(validation, validation_path)
    save_lines_gz((record.to_rsmi_line() for record in result.records), all_rsmi_path)
    save_lines_gz((record.to_rsmi_line() for record in training), training_rsmi_path)
    save_lines_gz((record.to_rsmi_line() for record in validation), validation_rsmi_path)
    manifest = create_manifest(
        action=TRAINING_REACTION_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            ("all", all_path, result.records, ContentType.UNKNOWN),
            ("training", training_path, training, ContentType.UNKNOWN),
            ("validation", validation_path, validation, ContentType.UNKNOWN),
        ],
        root_dir=source_root,
        parameters=config.to_manifest_dict(),
        statistics=result.summary.get("output", {}),
        summary=result.summary,
        release_name=result.release_name,
        keyed_output_files=True,
    )
    output_files = manifest.output_files
    if not isinstance(output_files, dict):
        raise TypeError("training reaction release manifest must use keyed output files")
    output_files["all"].content_hash = reaction_records_content_hash(result.records)
    output_files["training"].content_hash = reaction_records_content_hash(training)
    output_files["validation"].content_hash = reaction_records_content_hash(validation)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def summarize_reaction_records(records: Sequence[TrainingReactionRecord]) -> dict[str, Any]:
    training = sum(1 for record in records if record.split == "training")
    validation = sum(1 for record in records if record.split == "validation")
    return {"all_records": {"total": len(records), "training": training, "validation": validation}}


def reaction_records_content_hash(records: Sequence[TrainingReactionRecord]) -> str:
    signatures = sorted(reaction_record_signature(record) for record in records)
    payload = json.dumps(signatures, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def reaction_record_signature(record: TrainingReactionRecord) -> tuple[Any, ...]:
    return (
        tuple(sorted(str(reactant) for reactant in record.reactants)),
        str(record.product),
        _condition_identity(record),
    )


def _merge_exact_reaction_duplicates(
    records: Sequence[TrainingReactionRecord],
) -> tuple[list[TrainingReactionRecord], int]:
    by_exact: dict[tuple[Any, ...], TrainingReactionRecord] = {}
    duplicates_removed = 0
    for record in records:
        exact_key = _reaction_exact_key(record)
        existing = by_exact.get(exact_key)
        if existing is None:
            by_exact[exact_key] = record.model_copy(deep=True)
            continue
        existing.sources.extend(record.sources)
        existing.alternative_mapped_smiles = sorted(
            {*(existing.alternative_mapped_smiles), *(record.alternative_mapped_smiles)} - {existing.mapped_smiles}
        )
        duplicates_removed += 1
    return list(by_exact.values()), duplicates_removed


def _merge_transform_equivalent_reactions(
    records: Sequence[TrainingReactionRecord],
) -> tuple[list[TrainingReactionRecord], int]:
    grouped: dict[tuple[Any, ...], list[TrainingReactionRecord]] = defaultdict(list)
    for record in records:
        identity = _reaction_identity(record)
        grouped[identity].append(record)

    output = []
    duplicates_removed = 0
    for group in grouped.values():
        duplicates_removed += len(group) - 1
        if len(group) == 1:
            output.append(group[0])
            continue

        mapped_weights: dict[ReactionSmilesStr, int] = defaultdict(int)
        all_mapped: set[ReactionSmilesStr] = set()
        for record in group:
            mapped_weights[record.mapped_smiles] += len(record.sources)
            all_mapped.add(record.mapped_smiles)
            all_mapped.update(record.alternative_mapped_smiles)
        canonical_mapped = min(mapped_weights, key=lambda mapped: (-mapped_weights[mapped], mapped))
        canonical = min(
            (record for record in group if record.mapped_smiles == canonical_mapped),
            key=lambda record: tuple(source.route_id for source in record.sources),
        )
        output.append(
            canonical.model_copy(
                update={
                    "alternative_mapped_smiles": sorted(mapped for mapped in all_mapped if mapped != canonical_mapped),
                    "sources": [source for record in group for source in record.sources],
                },
                deep=True,
            )
        )
    return output, duplicates_removed


def _reaction_exact_key(record: TrainingReactionRecord) -> tuple[Any, ...]:
    return (record.mapped_smiles, tuple(record.condition_slot_smiles), record.condition_slot)


def _reaction_identity(record: TrainingReactionRecord) -> tuple[Any, ...]:
    return reaction_record_signature(record)


def _condition_identity(record: TrainingReactionRecord) -> tuple[str, ...]:
    condition_smiles = tuple(str(value) for value in record.condition_slot_smiles)
    if condition_smiles:
        return condition_smiles
    return (record.condition_slot,) if record.condition_slot is not None else ()


def _summarize_cross_split_overlap(
    training: Sequence[TrainingReactionRecord], validation: Sequence[TrainingReactionRecord]
) -> dict[str, int]:
    training_exact = {_reaction_exact_key(record) for record in training}
    validation_exact = {_reaction_exact_key(record) for record in validation}
    training_identities = {_reaction_identity(record) for record in training}
    validation_identities = {_reaction_identity(record) for record in validation}
    shared_identities = training_identities & validation_identities
    return {
        "shared_exact_reaction_signatures": len(training_exact & validation_exact),
        "shared_reaction_identities": len(shared_identities),
        "training_records_with_shared_identity": sum(
            1 for record in training if _reaction_identity(record) in shared_identities
        ),
        "validation_records_with_shared_identity": sum(
            1 for record in validation if _reaction_identity(record) in shared_identities
        ),
    }


def _renumber(records: Sequence[TrainingReactionRecord], *, prefix: str) -> list[TrainingReactionRecord]:
    output = []
    width = max(6, len(str(len(records))))
    for index, record in enumerate(records, start=1):
        output.append(record.model_copy(update={"id": f"{prefix}-rxn-{index:0{width}d}"}))
    return output
