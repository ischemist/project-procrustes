from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import (
    PreparedTrainingReaction,
    ReactionIdentityKey,
    SplitName,
    TrainingReactionBuildResult,
    TrainingReactionRecord,
    TrainingReactionSource,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
)
from retrocast.curation.training.route_release import (
    build_training_manifest,
    get_step_condition_slot_smiles,
    split_counts,
)
from retrocast.io import save_jsonl_gz, save_lines_gz
from retrocast.typing import ReactionSmilesStr

logger = logging.getLogger(__name__)
TRAINING_REACTION_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-single-step-release"


def get_training_reaction_release_name(config: TrainingSetBuildConfig) -> str:
    return f"single-step-{config.release_name}"


def get_training_reaction_record_id(config: TrainingSetBuildConfig, ordinal: int) -> str:
    return f"{config.route_prefix}-rxn-{get_training_reaction_release_name(config)}-{ordinal:06d}"


def get_exact_reaction_signature(reaction: PreparedTrainingReaction) -> str:
    """Identity for unquestionably identical flat reactions."""
    payload = {
        "mapped_smiles": reaction.mapped_smiles,
        "condition_slot_smiles": list(reaction.condition_slot_smiles),
        "condition_slot": reaction.condition_slot,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def get_reaction_identity_key(reaction: PreparedTrainingReaction) -> ReactionIdentityKey:
    """Identity for flat-reaction leakage checks and transform-agnostic grouping."""
    return (
        reaction.reactants,
        reaction.product,
        reaction.condition_slot_smiles or reaction.condition_slot,
    )


def flatten_training_route_records_to_reactions(
    route_records: Sequence[TrainingRouteRecord],
) -> list[PreparedTrainingReaction]:
    """Convert released route records into flat reaction candidates while preserving route lineage."""
    flattened_reactions: list[PreparedTrainingReaction] = []

    for route_record in route_records:
        for step_index, (product, step) in enumerate(route_record.route.iter_reactions(), start=1):
            if step.mapped_smiles is None:
                raise ValueError(
                    f"single-step release requires mapped_smiles; missing on route {route_record.id} step {step_index}"
                )

            condition_slot = step.metadata.get("condition_slot")
            condition_slot_str = condition_slot if isinstance(condition_slot, str) and condition_slot else None
            alternative_mapped_smiles = step.metadata.get("alternative_mapped_smiles")
            source = TrainingReactionSource(
                route_id=route_record.id,
                step_index=step_index,
                source_id=step.metadata.get("source_id") if isinstance(step.metadata.get("source_id"), str) else None,
                dataset=route_record.sources[0].dataset if route_record.sources else "unknown",
                raw_route_indices=[source.raw_index for source in route_record.sources],
                raw_route_hashes=[source.raw_route_hash for source in route_record.sources],
                patent_ids=[source.patent_id for source in route_record.sources],
            )
            flattened_reactions.append(
                PreparedTrainingReaction(
                    reactants=tuple(sorted(reactant.smiles for reactant in step.reactants)),
                    product=product.smiles,
                    mapped_smiles=step.mapped_smiles,
                    alternative_mapped_smiles=sorted(
                        mapped_smiles
                        for mapped_smiles in (alternative_mapped_smiles or [])
                        if isinstance(mapped_smiles, str)
                    ),
                    condition_slot=condition_slot_str,
                    condition_slot_smiles=get_step_condition_slot_smiles(step),
                    sources=[source],
                )
            )

    return flattened_reactions


def merge_exact_reaction_duplicates(
    reactions: Sequence[PreparedTrainingReaction],
) -> tuple[list[PreparedTrainingReaction], int]:
    """Collapse exact duplicate flat reactions."""
    reactions_by_signature: dict[str, PreparedTrainingReaction] = {}
    duplicates_removed = 0

    for reaction in reactions:
        exact_signature = get_exact_reaction_signature(reaction)
        existing_reaction = reactions_by_signature.get(exact_signature)
        if existing_reaction is None:
            reactions_by_signature[exact_signature] = PreparedTrainingReaction(
                reactants=reaction.reactants,
                product=reaction.product,
                mapped_smiles=reaction.mapped_smiles,
                alternative_mapped_smiles=sorted(
                    {
                        mapped_smiles
                        for mapped_smiles in reaction.alternative_mapped_smiles
                        if mapped_smiles != reaction.mapped_smiles
                    }
                ),
                condition_slot=reaction.condition_slot,
                condition_slot_smiles=reaction.condition_slot_smiles,
                transform_id=reaction.transform_id,
                sources=list(reaction.sources),
            )
            continue

        existing_reaction.alternative_mapped_smiles = sorted(
            {
                *existing_reaction.alternative_mapped_smiles,
                *reaction.alternative_mapped_smiles,
            }
            - {existing_reaction.mapped_smiles}
        )
        existing_reaction.sources.extend(reaction.sources)
        duplicates_removed += 1

    return list(reactions_by_signature.values()), duplicates_removed


def merge_transform_equivalent_reactions(
    reactions: Sequence[PreparedTrainingReaction],
) -> tuple[list[PreparedTrainingReaction], int]:
    """Collapse mapped-smiles variants for flat reactions."""
    grouped_reactions: dict[tuple[Any, ...], list[PreparedTrainingReaction]] = defaultdict(list)
    for reaction in reactions:
        grouped_reactions[(*get_reaction_identity_key(reaction), reaction.transform_id)].append(reaction)

    merged_reactions: list[PreparedTrainingReaction] = []
    duplicates_removed = 0

    for group in grouped_reactions.values():
        duplicates_removed += len(group) - 1
        if len(group) == 1:
            merged_reactions.append(group[0])
            continue

        mapped_smiles_weights: dict[ReactionSmilesStr, int] = defaultdict(int)
        all_mapped_smiles: set[ReactionSmilesStr] = set()
        for reaction in group:
            mapped_smiles_weights[reaction.mapped_smiles] += len(reaction.sources)
            all_mapped_smiles.add(reaction.mapped_smiles)
            all_mapped_smiles.update(reaction.alternative_mapped_smiles)

        canonical_mapped_smiles = min(
            mapped_smiles_weights,
            key=lambda mapped_smiles: (-mapped_smiles_weights[mapped_smiles], mapped_smiles),
        )
        canonical_reaction = min(
            (reaction for reaction in group if reaction.mapped_smiles == canonical_mapped_smiles),
            key=lambda reaction: tuple(source.route_id for source in reaction.sources),
        )
        merged_reactions.append(
            PreparedTrainingReaction(
                reactants=canonical_reaction.reactants,
                product=canonical_reaction.product,
                mapped_smiles=canonical_mapped_smiles,
                alternative_mapped_smiles=sorted(
                    mapped_smiles for mapped_smiles in all_mapped_smiles if mapped_smiles != canonical_mapped_smiles
                ),
                condition_slot=canonical_reaction.condition_slot,
                condition_slot_smiles=canonical_reaction.condition_slot_smiles,
                transform_id=canonical_reaction.transform_id,
                sources=[source for reaction in group for source in reaction.sources],
            )
        )

    return merged_reactions, duplicates_removed


def summarize_reaction_records(records: Sequence[TrainingReactionRecord]) -> dict[str, Any]:
    training_records = [record for record in records if record.split == "training"]
    validation_records = [record for record in records if record.split == "validation"]

    return {
        "all_records": split_counts(
            total=len(records),
            training=len(training_records),
            validation=len(validation_records),
        ),
    }


def summarize_cross_split_reaction_overlap(
    *,
    training_reactions: Sequence[PreparedTrainingReaction],
    validation_reactions: Sequence[PreparedTrainingReaction],
) -> dict[str, int]:
    training_exact_signatures = {get_exact_reaction_signature(reaction) for reaction in training_reactions}
    validation_exact_signatures = {get_exact_reaction_signature(reaction) for reaction in validation_reactions}
    training_identity_keys = {get_reaction_identity_key(reaction) for reaction in training_reactions}
    validation_identity_keys = {get_reaction_identity_key(reaction) for reaction in validation_reactions}
    shared_identity_keys = training_identity_keys & validation_identity_keys

    return {
        "shared_exact_reaction_signatures": len(training_exact_signatures & validation_exact_signatures),
        "shared_reaction_identities": len(shared_identity_keys),
        "training_records_with_shared_identity": sum(
            1 for reaction in training_reactions if get_reaction_identity_key(reaction) in shared_identity_keys
        ),
        "validation_records_with_shared_identity": sum(
            1 for reaction in validation_reactions if get_reaction_identity_key(reaction) in shared_identity_keys
        ),
    }


def drop_cross_split_validation_overlap(
    *,
    training_reactions: Sequence[PreparedTrainingReaction],
    validation_reactions: Sequence[PreparedTrainingReaction],
) -> tuple[list[PreparedTrainingReaction], int]:
    """Remove validation reactions whose identity already appears in training."""
    training_identity_keys = {get_reaction_identity_key(reaction) for reaction in training_reactions}
    filtered_validation_reactions = [
        reaction
        for reaction in validation_reactions
        if get_reaction_identity_key(reaction) not in training_identity_keys
    ]
    return filtered_validation_reactions, len(validation_reactions) - len(filtered_validation_reactions)


def build_training_reaction_records_from_route_records(
    route_records: Sequence[TrainingRouteRecord],
    config: TrainingSetBuildConfig,
) -> TrainingReactionBuildResult:
    if config.holdout_mode != "reaction":
        raise ValueError("single-step training release requires TrainingSetBuildConfig(holdout_mode='reaction')")

    route_records_by_split: dict[SplitName, list[TrainingRouteRecord]] = {
        "training": [],
        "validation": [],
    }
    for route_record in route_records:
        route_records_by_split[route_record.split].append(route_record)

    split_reactions: dict[SplitName, list[PreparedTrainingReaction]] = {}
    split_postprocessing: dict[SplitName, dict[str, int]] = {}
    for split in ("training", "validation"):
        flattened_reactions = flatten_training_route_records_to_reactions(route_records_by_split[split])
        chemically_unique_reactions, exact_chemical_duplicates_removed = merge_exact_reaction_duplicates(
            flattened_reactions
        )
        deduplicated_reactions, mapped_smiles_variants_collapsed = merge_transform_equivalent_reactions(
            chemically_unique_reactions
        )
        split_reactions[split] = deduplicated_reactions
        split_postprocessing[split] = {
            "input_routes": len(route_records_by_split[split]),
            "flattened_reactions": len(flattened_reactions),
            "chemical_duplicates_removed": exact_chemical_duplicates_removed,
            "mapped_smiles_variants_collapsed": mapped_smiles_variants_collapsed,
            "duplicate_reactions_removed": exact_chemical_duplicates_removed + mapped_smiles_variants_collapsed,
        }

    overlap_before_cleanup = summarize_cross_split_reaction_overlap(
        training_reactions=split_reactions["training"],
        validation_reactions=split_reactions["validation"],
    )
    cleaned_validation_reactions, overlap_removed_from_validation = drop_cross_split_validation_overlap(
        training_reactions=split_reactions["training"],
        validation_reactions=split_reactions["validation"],
    )
    split_reactions["validation"] = cleaned_validation_reactions
    overlap_after_cleanup = summarize_cross_split_reaction_overlap(
        training_reactions=split_reactions["training"],
        validation_reactions=split_reactions["validation"],
    )
    if overlap_after_cleanup["shared_reaction_identities"] != 0:
        raise ValueError("single-step release validation split still overlaps with training after cleanup")
    if overlap_after_cleanup["shared_exact_reaction_signatures"] != 0:
        raise ValueError(
            "single-step release validation split still shares exact reactions with training after cleanup"
        )

    records: list[TrainingReactionRecord] = []
    for split in ("training", "validation"):
        for reaction in split_reactions[split]:
            records.append(
                TrainingReactionRecord(
                    id="",
                    split=split,
                    reactants=list(reaction.reactants),
                    product=reaction.product,
                    mapped_smiles=reaction.mapped_smiles,
                    alternative_mapped_smiles=list(reaction.alternative_mapped_smiles),
                    condition_slot=reaction.condition_slot,
                    condition_slot_smiles=list(reaction.condition_slot_smiles),
                    sources=list(reaction.sources),
                )
            )

    for ordinal, record in enumerate(records, start=1):
        record.id = get_training_reaction_record_id(config, ordinal)

    summary = {
        "input": {
            "route_records": split_counts(
                total=len(route_records),
                training=len(route_records_by_split["training"]),
                validation=len(route_records_by_split["validation"]),
            )
        },
        "reaction_postprocessing": {
            "training": split_postprocessing["training"],
            "validation": {
                **split_postprocessing["validation"],
                "overlap_removed_from_validation": overlap_removed_from_validation,
            },
            "cross_split_overlap_before_cleanup": overlap_before_cleanup,
            "cross_split_overlap_after_cleanup": overlap_after_cleanup,
            "dedup_contract": {
                "exact_duplicates": "collapse identical mapped_smiles + condition identity within each split",
                "mapping_drift": (
                    "collapse reactants + product + condition identity within each split; "
                    "paroutes transform ids are unavailable in the released route artifact"
                ),
                "validation_cleanup": (
                    "drop validation reactions whose reactants + product + condition identity already appears in training"
                ),
            },
        },
        "output": summarize_reaction_records(records),
    }

    return TrainingReactionBuildResult(
        release_name=get_training_reaction_release_name(config),
        records=records,
        summary=summary,
    )


def write_training_reaction_release(
    result: TrainingReactionBuildResult,
    output_dir: Path,
    source_paths: Sequence[Path],
    config: TrainingSetBuildConfig,
    source_root: Path | None = None,
) -> dict[str, Any]:
    release_dir = output_dir / result.release_name
    release_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing single-step training release to %s", release_dir)

    files = {
        "all": release_dir / "all.jsonl.gz",
        "training": release_dir / "training.jsonl.gz",
        "validation": release_dir / "validation.jsonl.gz",
        "all_rsmi": release_dir / "all.rsmi.txt.gz",
        "training_rsmi": release_dir / "training.rsmi.txt.gz",
        "validation_rsmi": release_dir / "validation.rsmi.txt.gz",
    }
    grouped_records = {
        "all": result.records,
        "training": [record for record in result.records if record.split == "training"],
        "validation": [record for record in result.records if record.split == "validation"],
    }
    for name, records in grouped_records.items():
        logger.info("  writing %s (%s records)", files[name].name, f"{len(records):,}")
        save_jsonl_gz(records, files[name])
        rsmi_key = f"{name}_rsmi"
        logger.info("  writing %s (%s lines)", files[rsmi_key].name, f"{len(records):,}")
        save_lines_gz((record.to_rsmi_line() for record in records), files[rsmi_key])

    manifest = build_training_manifest(
        release_name=result.release_name,
        files=files,
        source_paths=source_paths,
        source_root=source_root,
        config=config,
        summary=result.summary,
        action=TRAINING_REACTION_RELEASE_ACTION,
    )
    manifest_path = release_dir / "manifest.json"
    logger.info("  writing %s", manifest_path.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
