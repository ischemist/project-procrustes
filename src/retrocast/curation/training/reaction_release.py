from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import (
    ReactionIdentityKey,
    SplitName,
    TrainingReactionBuildResult,
    TrainingReactionCandidate,
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
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import save_jsonl_gz, save_lines_gz
from retrocast.typing import ReactionSmilesStr

logger = logging.getLogger(__name__)
TRAINING_REACTION_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-single-step-release"
ReactionExactKey = tuple[ReactionSmilesStr, tuple[str, ...], str | None]


def exact_reaction_key(reaction: TrainingReactionCandidate) -> ReactionExactKey:
    """Identity for unquestionably identical flat reactions."""
    return (reaction.mapped_smiles, tuple(reaction.condition_slot_smiles), reaction.condition_slot)


def reaction_identity_key(reaction: TrainingReactionCandidate) -> ReactionIdentityKey:
    """Identity for flat-reaction leakage checks and transform-agnostic grouping."""
    return (
        reaction.reactants,
        reaction.product,
        reaction.condition_slot_smiles or reaction.condition_slot,
    )


def flatten_training_route_records_to_reactions(
    route_records: Sequence[TrainingRouteRecord],
) -> list[TrainingReactionCandidate]:
    """Convert released route records into flat reaction candidates while preserving route lineage."""
    flattened_reactions: list[TrainingReactionCandidate] = []

    for route_record in route_records:
        for step_index, route_reaction in enumerate(route_record.route.iter_reactions(), start=1):
            product = route_reaction.product
            step = route_reaction.step
            if step.mapped_smiles is None:
                raise TrainingReleaseError(
                    f"single-step release requires mapped_smiles; missing on route {route_record.id} step {step_index}",
                    code="workflow.single_step_missing_mapped_smiles",
                    context={"route_id": route_record.id, "step_index": step_index},
                )

            condition_slot = step.metadata.get("condition_slot")
            condition_slot_str = condition_slot if isinstance(condition_slot, str) and condition_slot else None
            alternative_mapped_smiles = step.metadata.get("alternative_mapped_smiles")
            source = TrainingReactionSource(
                route_id=route_record.id,
                step_index=step_index,
                source_id=step.metadata.get("source_id") if isinstance(step.metadata.get("source_id"), str) else None,
            )
            flattened_reactions.append(
                TrainingReactionCandidate(
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
    reactions: Sequence[TrainingReactionCandidate],
) -> tuple[list[TrainingReactionCandidate], int]:
    """Collapse exact duplicate flat reactions."""
    reactions_by_signature: dict[ReactionExactKey, TrainingReactionCandidate] = {}
    duplicates_removed = 0

    for reaction in reactions:
        exact_signature = exact_reaction_key(reaction)
        existing_reaction = reactions_by_signature.get(exact_signature)
        if existing_reaction is None:
            reactions_by_signature[exact_signature] = TrainingReactionCandidate(
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
    reactions: Sequence[TrainingReactionCandidate],
) -> tuple[list[TrainingReactionCandidate], int]:
    """Collapse mapped-smiles variants for flat reactions."""
    grouped_reactions: dict[ReactionIdentityKey, list[TrainingReactionCandidate]] = defaultdict(list)
    for reaction in reactions:
        grouped_reactions[reaction_identity_key(reaction)].append(reaction)

    merged_reactions: list[TrainingReactionCandidate] = []
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
            TrainingReactionCandidate(
                reactants=canonical_reaction.reactants,
                product=canonical_reaction.product,
                mapped_smiles=canonical_mapped_smiles,
                alternative_mapped_smiles=sorted(
                    mapped_smiles for mapped_smiles in all_mapped_smiles if mapped_smiles != canonical_mapped_smiles
                ),
                condition_slot=canonical_reaction.condition_slot,
                condition_slot_smiles=canonical_reaction.condition_slot_smiles,
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


@dataclass
class TrainingReactionReleaseBuilder:
    route_records: Sequence[TrainingRouteRecord]
    config: TrainingSetBuildConfig
    route_records_by_split: dict[SplitName, list[TrainingRouteRecord]] = dataclass_field(init=False)
    split_reactions: dict[SplitName, list[TrainingReactionCandidate]] = dataclass_field(
        default_factory=dict, init=False
    )
    split_postprocessing: dict[SplitName, dict[str, int]] = dataclass_field(default_factory=dict, init=False)
    overlap_before_cleanup: dict[str, int] = dataclass_field(default_factory=dict, init=False)
    overlap_after_cleanup: dict[str, int] = dataclass_field(default_factory=dict, init=False)
    overlap_removed_from_validation: int = dataclass_field(default=0, init=False)
    _started: bool = dataclass_field(default=False, init=False)

    def __post_init__(self) -> None:
        self.route_records_by_split = {"training": [], "validation": []}

    def build(self) -> TrainingReactionBuildResult:
        self.assert_valid_config()
        self.assert_not_started()
        self._started = True
        self.group_route_records_by_split()
        for split in ("training", "validation"):
            self.build_split(split)
        self.remove_validation_overlap()
        records = self.materialize_records()
        return TrainingReactionBuildResult(
            release_name=f"single-step-{self.config.release_name}",
            records=records,
            summary={**self.summary(), "output": summarize_reaction_records(records)},
        )

    def assert_valid_config(self) -> None:
        if self.config.holdout_mode != "reaction":
            raise TrainingReleaseError(
                "single-step training release requires TrainingSetBuildConfig(holdout_mode='reaction')",
                code="workflow.single_step_requires_reaction_holdout",
                context={"holdout_mode": self.config.holdout_mode},
            )

    def assert_not_started(self) -> None:
        if (
            self._started
            or self.route_records_by_split["training"]
            or self.route_records_by_split["validation"]
            or self.split_reactions
            or self.split_postprocessing
            or self.overlap_before_cleanup
            or self.overlap_after_cleanup
            or self.overlap_removed_from_validation
        ):
            raise RuntimeError("TrainingReactionReleaseBuilder instances are single-use")

    def group_route_records_by_split(self) -> None:
        for route_record in self.route_records:
            self.route_records_by_split[route_record.split].append(route_record)

    def build_split(self, split: SplitName) -> None:
        """Flatten and deduplicate one split without looking across split boundaries."""
        flattened_reactions = flatten_training_route_records_to_reactions(self.route_records_by_split[split])
        chemically_unique_reactions, exact_duplicates_removed = merge_exact_reaction_duplicates(flattened_reactions)
        deduplicated_reactions, mapped_variants_collapsed = merge_transform_equivalent_reactions(
            chemically_unique_reactions
        )
        self.split_reactions[split] = deduplicated_reactions
        self.split_postprocessing[split] = {
            "input_routes": len(self.route_records_by_split[split]),
            "flattened_reactions": len(flattened_reactions),
            "chemical_duplicates_removed": exact_duplicates_removed,
            "mapped_smiles_variants_collapsed": mapped_variants_collapsed,
            "duplicate_reactions_removed": exact_duplicates_removed + mapped_variants_collapsed,
        }

    def remove_validation_overlap(self) -> None:
        """Make validation strict by dropping reactions whose identity appears in training."""
        self.overlap_before_cleanup = self.summarize_cross_split_overlap()
        training_keys = {reaction_identity_key(reaction) for reaction in self.split_reactions["training"]}
        validation_reactions = self.split_reactions["validation"]
        self.split_reactions["validation"] = [
            reaction for reaction in validation_reactions if reaction_identity_key(reaction) not in training_keys
        ]
        self.overlap_removed_from_validation = len(validation_reactions) - len(self.split_reactions["validation"])
        self.overlap_after_cleanup = self.summarize_cross_split_overlap()
        self.assert_no_cross_split_overlap()

    def summarize_cross_split_overlap(self) -> dict[str, int]:
        training_reactions = self.split_reactions["training"]
        validation_reactions = self.split_reactions["validation"]
        training_exact_keys = {exact_reaction_key(reaction) for reaction in training_reactions}
        validation_exact_keys = {exact_reaction_key(reaction) for reaction in validation_reactions}
        training_identity_keys = {reaction_identity_key(reaction) for reaction in training_reactions}
        validation_identity_keys = {reaction_identity_key(reaction) for reaction in validation_reactions}
        shared_identity_keys = training_identity_keys & validation_identity_keys

        return {
            "shared_exact_reaction_signatures": len(training_exact_keys & validation_exact_keys),
            "shared_reaction_identities": len(shared_identity_keys),
            "training_records_with_shared_identity": sum(
                1 for reaction in training_reactions if reaction_identity_key(reaction) in shared_identity_keys
            ),
            "validation_records_with_shared_identity": sum(
                1 for reaction in validation_reactions if reaction_identity_key(reaction) in shared_identity_keys
            ),
        }

    def assert_no_cross_split_overlap(self) -> None:
        if self.overlap_after_cleanup["shared_reaction_identities"] != 0:
            raise TrainingReleaseError(
                "single-step release validation split still overlaps with training after cleanup",
                code="workflow.single_step_validation_overlap",
                context=self.overlap_after_cleanup,
            )
        if self.overlap_after_cleanup["shared_exact_reaction_signatures"] != 0:
            raise TrainingReleaseError(
                "single-step release validation split still shares exact reactions with training after cleanup",
                code="workflow.single_step_exact_validation_overlap",
                context=self.overlap_after_cleanup,
            )

    def materialize_records(self) -> list[TrainingReactionRecord]:
        """Attach release ids and split names after all deduplication and cleanup are final."""
        records: list[TrainingReactionRecord] = []
        for split in ("training", "validation"):
            for reaction in self.split_reactions[split]:
                records.append(
                    TrainingReactionRecord(
                        id=f"{self.config.route_prefix}-rxn-{len(records) + 1:06d}",
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
        return records

    def summary(self) -> dict[str, Any]:
        return {
            "input": {
                "parent_route_release": self.config.release_name,
                "route_records": split_counts(
                    total=len(self.route_records),
                    training=len(self.route_records_by_split["training"]),
                    validation=len(self.route_records_by_split["validation"]),
                ),
                "provenance": (
                    "reaction sources store route_id, step_index, and optional source_id; "
                    "raw route hashes and patent ids live in the parent route release"
                ),
            },
            "reaction_postprocessing": {
                "training": self.split_postprocessing["training"],
                "validation": {
                    **self.split_postprocessing["validation"],
                    "overlap_removed_from_validation": self.overlap_removed_from_validation,
                },
                "cross_split_overlap_before_cleanup": self.overlap_before_cleanup,
                "cross_split_overlap_after_cleanup": self.overlap_after_cleanup,
                "dedup_contract": {
                    "exact_duplicates": "collapse identical mapped_smiles + condition identity within each split",
                    "mapping_drift": ("collapse reactants + product + condition identity within each split"),
                    "validation_cleanup": (
                        "drop validation reactions whose reactants + product + condition identity already appears in training"
                    ),
                },
            },
        }


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
