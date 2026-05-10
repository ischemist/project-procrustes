from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from tqdm.auto import tqdm

from retrocast.adapters import adapt_single_route_with_diagnostics
from retrocast.curation.filtering import deduplicate_routes, excise_reactions_from_route
from retrocast.io import ContentType, create_manifest, save_jsonl_gz
from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route, TargetInput

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["train", "val"]
ConditionIdentity = tuple[str, ...] | str | None
TransformDedupStepKey = tuple[str | None, ConditionIdentity]
TransformDedupKey = tuple[str, tuple[TransformDedupStepKey, ...]]
T = TypeVar("T")

logger = logging.getLogger(__name__)
TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"


@dataclass(frozen=True)
class RawRouteSource:
    dataset: str
    raw_index: int
    raw_route_hash: str
    patent_id: str | None = None


@dataclass
class TrainingRouteRecord:
    id: str
    split: SplitName
    route_signature: str
    content_hash: str
    route: Route
    sources: list[RawRouteSource] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "route_signature": self.route_signature,
            "content_hash": self.content_hash,
            "source": {
                "dataset": "paroutes-all-routes",
                "raw_indices": [source.raw_index for source in self.sources],
                "raw_route_hashes": [source.raw_route_hash for source in self.sources],
                "patent_ids": [source.patent_id for source in self.sources],
            },
            "route": self.route.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class TrainingSetBuildConfig:
    holdout_mode: TrainingHoldoutMode
    val_fraction: float = 0.05
    seed: int = 20260502
    route_prefix: str = "paroutes"
    show_progress: bool = True

    @property
    def release_name(self) -> str:
        return f"{self.holdout_mode}-heldout-n1-n5"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "holdout_mode": self.holdout_mode,
            "split": {
                "val_fraction": self.val_fraction,
                "seed": self.seed,
            },
            "progress": {
                "enabled": self.show_progress,
            },
            "release_rules": {
                "route_holdout": "exclude full n1 union n5 by route.get_structural_signature()",
                "reaction_holdout": "excise reactions present in n1 union n5 and keep surviving sub-routes",
                "chemical_exact_dedup": "collapse exact route duplicates by route.get_annotated_signature(include_mapped_smiles=True)",
                "transform_dedup": "collapse mapped-smiles variants by structure + condition identity + raw paroutes reaction_hash sidecar",
            },
        }


@dataclass
class TrainingSetBuildResult:
    release_name: str
    records: list[TrainingRouteRecord]
    summary: dict[str, Any]


@dataclass(frozen=True)
class AdaptedTrainingRoute:
    route: Route
    structural_signature: str
    reaction_signatures: set[ReactionSignature]
    source: RawRouteSource
    transform_ids_by_source_id: dict[str, str]


@dataclass
class PreparedTrainingRoute:
    route: Route
    structural_signature: str
    sources: list[RawRouteSource] = field(default_factory=list)
    transform_ids_by_source_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptationStatistics:
    raw_routes: int
    adapted_routes: int
    skipped_routes: int
    skipped_without_error_code: int
    failures_by_code: dict[str, int]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "raw_routes": self.raw_routes,
            "adapted_routes": self.adapted_routes,
            "skipped_routes": self.skipped_routes,
            "skipped_without_error_code": self.skipped_without_error_code,
            "failures_by_code": dict(sorted(self.failures_by_code.items())),
        }


def stable_raw_route_hash(raw_route: dict[str, Any]) -> str:
    payload = json.dumps(raw_route, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_paroutes_transform_ids_by_source_id(raw_route: Mapping[str, Any]) -> dict[str, str]:
    """Collect raw paroutes reaction_hash values keyed by reaction source id."""
    transform_ids_by_source_id: dict[str, str] = {}

    def _visit(node: Mapping[str, Any]) -> None:
        metadata = node.get("metadata")
        if isinstance(metadata, Mapping):
            source_id = metadata.get("ID")
            transform_id = metadata.get("reaction_hash")
            if isinstance(source_id, str) and isinstance(transform_id, str) and transform_id:
                transform_ids_by_source_id[source_id] = transform_id

        children = node.get("children")
        if not isinstance(children, list):
            return

        for child in children:
            if isinstance(child, Mapping):
                _visit(child)

    _visit(raw_route)
    return transform_ids_by_source_id


def iter_route_steps(route: Route) -> list[ReactionStep]:
    """Return reaction steps in deterministic root-first depth-first order."""
    steps: list[ReactionStep] = []

    def _visit(node: Molecule) -> None:
        if node.synthesis_step is None:
            return
        steps.append(node.synthesis_step)
        for reactant in node.synthesis_step.reactants:
            _visit(reactant)

    _visit(route.target)
    return steps


def get_exact_chemical_signature(route: Route) -> str:
    """Identity for unquestionably identical chemistry, excluding route provenance."""
    return route.get_annotated_signature(include_mapped_smiles=True)


def get_step_condition_identity(step: ReactionStep) -> ConditionIdentity:
    """Condition identity used for transform-level deduplication."""
    condition_slot_smiles = step.metadata.get("condition_slot_smiles")
    if isinstance(condition_slot_smiles, list) and all(isinstance(value, str) for value in condition_slot_smiles):
        return tuple(condition_slot_smiles)

    condition_slot = step.metadata.get("condition_slot")
    if isinstance(condition_slot, str) and condition_slot:
        return condition_slot

    return None


def get_transform_dedup_key(route: Route, *, transform_ids_by_source_id: Mapping[str, str]) -> TransformDedupKey:
    """Identity for collapsing mapped-smiles drift while preserving condition variants."""
    step_keys: list[TransformDedupStepKey] = []
    for step in iter_route_steps(route):
        source_id = step.metadata.get("source_id")
        transform_id = transform_ids_by_source_id.get(source_id) if isinstance(source_id, str) else None
        step_keys.append((transform_id, get_step_condition_identity(step)))
    return route.get_structural_signature(), tuple(step_keys)


def get_mapped_smiles_profile(route: Route) -> tuple[str | None, ...]:
    """Return the per-step mapped reaction smiles profile for canonical route selection."""
    return tuple(step.mapped_smiles for step in iter_route_steps(route))


def clone_route_for_training_release(route: Route) -> Route:
    """Clone a route and strip source-specific metadata that should not survive release merges."""
    cloned_route = route.model_copy(deep=True)
    cloned_route.metadata = cloned_route.metadata.copy()
    cloned_route.metadata.pop("patent_id", None)
    return cloned_route


def merge_alternative_mapped_smiles(canonical_route: Route, routes: Sequence[Route]) -> None:
    """Record non-canonical mapped-smiles variants on the kept route."""
    canonical_steps = iter_route_steps(canonical_route)
    route_steps = [iter_route_steps(route) for route in routes]
    for step_index, canonical_step in enumerate(canonical_steps):
        variant_smiles: set[str] = set()
        if canonical_step.mapped_smiles is not None:
            variant_smiles.add(canonical_step.mapped_smiles)

        existing_alternatives = canonical_step.metadata.get("alternative_mapped_smiles")
        if isinstance(existing_alternatives, list):
            variant_smiles.update(value for value in existing_alternatives if isinstance(value, str))

        for steps in route_steps:
            if len(steps) != len(canonical_steps):
                raise ValueError("cannot merge mapped smiles for routes with different step counts")

            mapped_smiles = steps[step_index].mapped_smiles
            if mapped_smiles is not None:
                variant_smiles.add(mapped_smiles)

            step_alternatives = steps[step_index].metadata.get("alternative_mapped_smiles")
            if isinstance(step_alternatives, list):
                variant_smiles.update(value for value in step_alternatives if isinstance(value, str))

        alternatives = sorted(
            mapped_smiles for mapped_smiles in variant_smiles if mapped_smiles != canonical_step.mapped_smiles
        )
        canonical_step.metadata = canonical_step.metadata.copy()
        if alternatives:
            canonical_step.metadata["alternative_mapped_smiles"] = alternatives
        else:
            canonical_step.metadata.pop("alternative_mapped_smiles", None)


def merge_exact_chemical_duplicates(routes: Sequence[AdaptedTrainingRoute]) -> tuple[list[PreparedTrainingRoute], int]:
    """Collapse exact duplicate chemistry while preserving all raw sources."""
    routes_by_signature: dict[str, PreparedTrainingRoute] = {}
    duplicates_removed = 0

    for adapted_route in routes:
        exact_signature = get_exact_chemical_signature(adapted_route.route)
        existing_route = routes_by_signature.get(exact_signature)
        if existing_route is None:
            routes_by_signature[exact_signature] = PreparedTrainingRoute(
                route=clone_route_for_training_release(adapted_route.route),
                structural_signature=adapted_route.structural_signature,
                sources=[adapted_route.source],
                transform_ids_by_source_id=dict(adapted_route.transform_ids_by_source_id),
            )
            continue

        existing_route.sources.append(adapted_route.source)
        duplicates_removed += 1

    return list(routes_by_signature.values()), duplicates_removed


def merge_transform_equivalent_routes(
    routes: Sequence[PreparedTrainingRoute],
) -> tuple[list[PreparedTrainingRoute], int]:
    """Collapse mapped-smiles variants that share structure, condition identity, and transform ids."""
    grouped_routes: dict[TransformDedupKey, list[PreparedTrainingRoute]] = defaultdict(list)
    for route in routes:
        grouped_routes[
            get_transform_dedup_key(route.route, transform_ids_by_source_id=route.transform_ids_by_source_id)
        ].append(route)

    merged_routes: list[PreparedTrainingRoute] = []
    duplicates_removed = 0

    for group in grouped_routes.values():
        duplicates_removed += len(group) - 1
        if len(group) == 1:
            merged_routes.append(group[0])
            continue

        profile_weights: dict[tuple[str | None, ...], int] = defaultdict(int)
        routes_by_profile: dict[tuple[str | None, ...], list[PreparedTrainingRoute]] = defaultdict(list)
        for route in group:
            profile = get_mapped_smiles_profile(route.route)
            profile_weights[profile] += len(route.sources)
            routes_by_profile[profile].append(route)

        canonical_profile = min(
            profile_weights,
            key=lambda profile: (
                -profile_weights[profile],
                tuple("" if value is None else value for value in profile),
            ),
        )
        canonical_source = min(
            routes_by_profile[canonical_profile],
            key=lambda route: tuple(source.raw_route_hash for source in route.sources),
        )

        merged_route = clone_route_for_training_release(canonical_source.route)
        merge_alternative_mapped_smiles(merged_route, [route.route for route in group])
        merged_routes.append(
            PreparedTrainingRoute(
                route=merged_route,
                structural_signature=canonical_source.structural_signature,
                sources=[source for route in group for source in route.sources],
                transform_ids_by_source_id=dict(canonical_source.transform_ids_by_source_id),
            )
        )

    return merged_routes, duplicates_removed


def progress_iter(items: Sequence[T], desc: str, enabled: bool) -> Iterable[T]:
    if not enabled:
        return items
    return tqdm(items, desc=desc, unit="route", dynamic_ncols=True, leave=False)


def adapt_training_routes(
    raw_routes: Sequence[dict[str, Any]],
    dataset: str,
    id_width: int,
    collect_reactions: bool,
    show_progress: bool = True,
) -> tuple[list[AdaptedTrainingRoute], AdaptationStatistics]:
    adapted_routes: list[AdaptedTrainingRoute] = []
    failures_by_code: dict[str, int] = defaultdict(int)
    skipped_adaptation = 0
    skipped_without_error_code = 0
    routes = progress_iter(raw_routes, desc=f"adapt {dataset}", enabled=show_progress)
    for idx, raw_route in enumerate(routes):
        target_id = f"{dataset}-{idx + 1:0{id_width}d}"
        raw_smiles = raw_route.get("smiles") if isinstance(raw_route, Mapping) else None
        if not isinstance(raw_smiles, str) or not raw_smiles:
            skipped_adaptation += 1
            failures_by_code["adapter.schema_invalid"] += 1
            continue
        target = TargetInput(id=target_id, smiles=raw_smiles)
        adaptation = adapt_single_route_with_diagnostics(raw_route, target, "paroutes")
        if adaptation.route is None:
            skipped_adaptation += 1
            if adaptation.error is None:
                skipped_without_error_code += 1
            else:
                failures_by_code[adaptation.error.code] += 1
            continue
        route = adaptation.route
        patent_id = route.metadata.get("patent_id") if isinstance(route.metadata.get("patent_id"), str) else None
        adapted_routes.append(
            AdaptedTrainingRoute(
                route=route,
                structural_signature=route.get_structural_signature(),
                reaction_signatures=route.get_reaction_signatures() if collect_reactions else set(),
                source=RawRouteSource(
                    dataset=dataset,
                    raw_index=idx,
                    raw_route_hash=stable_raw_route_hash(raw_route),
                    patent_id=patent_id,
                ),
                transform_ids_by_source_id=extract_paroutes_transform_ids_by_source_id(raw_route),
            )
        )
    return adapted_routes, AdaptationStatistics(
        raw_routes=len(raw_routes),
        adapted_routes=len(adapted_routes),
        skipped_routes=skipped_adaptation,
        skipped_without_error_code=skipped_without_error_code,
        failures_by_code=dict(failures_by_code),
    )


def collect_heldout_signatures(
    routes_by_dataset: Mapping[str, Sequence[AdaptedTrainingRoute]],
    collect_reactions: bool,
) -> tuple[set[str], set[ReactionSignature]]:
    route_signatures: set[str] = set()
    reaction_signatures: set[ReactionSignature] = set()
    for routes in routes_by_dataset.values():
        for route in routes:
            route_signatures.add(route.structural_signature)
            if collect_reactions:
                reaction_signatures.update(route.reaction_signatures)
    return route_signatures, reaction_signatures


def excise_heldout_reactions(
    adapted_route: AdaptedTrainingRoute,
    heldout_reaction_signatures: set[ReactionSignature],
) -> tuple[list[AdaptedTrainingRoute], bool]:
    """Excise heldout reactions from one route and return surviving unique fragments."""
    overlapping_signatures = adapted_route.reaction_signatures & heldout_reaction_signatures
    if not overlapping_signatures:
        return [adapted_route], False

    excised_routes = excise_reactions_from_route(adapted_route.route, overlapping_signatures)
    unique_excised_routes = deduplicate_routes(excised_routes)
    return [
        AdaptedTrainingRoute(
            route=route,
            structural_signature=route.get_structural_signature(),
            reaction_signatures=route.get_reaction_signatures(),
            source=adapted_route.source,
            transform_ids_by_source_id=dict(adapted_route.transform_ids_by_source_id),
        )
        for route in unique_excised_routes
    ], True


def build_training_records_from_adapted(
    all_routes: Sequence[AdaptedTrainingRoute],
    all_adaptation: AdaptationStatistics,
    heldout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
    heldout_adaptation: Mapping[str, AdaptationStatistics] | None,
    config: TrainingSetBuildConfig,
) -> TrainingSetBuildResult:
    heldout_route_signatures, heldout_reaction_signatures = collect_heldout_signatures(
        heldout_routes,
        collect_reactions=config.holdout_mode == "reaction",
    )

    candidate_routes: list[AdaptedTrainingRoute] = []
    skipped_route_holdout = 0
    reaction_excision_source_routes = 0
    reaction_excision_fragments = 0
    fully_removed_by_reaction_excision = 0

    for adapted_route in all_routes:
        if adapted_route.structural_signature in heldout_route_signatures:
            skipped_route_holdout += 1
            continue

        candidate_routes_for_route = [adapted_route]
        if config.holdout_mode == "reaction":
            candidate_routes_for_route, was_excised = excise_heldout_reactions(
                adapted_route, heldout_reaction_signatures
            )
            if was_excised:
                reaction_excision_source_routes += 1
                reaction_excision_fragments += len(candidate_routes_for_route)
                if not candidate_routes_for_route:
                    fully_removed_by_reaction_excision += 1

        candidate_routes.extend(candidate_routes_for_route)

    chemically_unique_routes, exact_chemical_duplicates_removed = merge_exact_chemical_duplicates(candidate_routes)
    deduplicated_routes, mapped_smiles_variants_collapsed = merge_transform_equivalent_routes(chemically_unique_routes)

    prepared_routes_by_key = {
        f"route-{ordinal:09d}": prepared_route for ordinal, prepared_route in enumerate(deduplicated_routes, start=1)
    }

    split_by_signature = assign_train_val_splits(
        {route_key: prepared_route.route for route_key, prepared_route in prepared_routes_by_key.items()},
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    records: list[TrainingRouteRecord] = []
    for ordinal, route_key in enumerate(sorted(prepared_routes_by_key), start=1):
        prepared_route = prepared_routes_by_key[route_key]
        route = prepared_route.route
        records.append(
            TrainingRouteRecord(
                id=f"{config.route_prefix}-{config.release_name}-{ordinal:06d}",
                split=split_by_signature[route_key],
                route_signature=prepared_route.structural_signature,
                content_hash=route.get_content_hash(),
                route=route,
                sources=prepared_route.sources,
            )
        )

    postprocessing: dict[str, Any] = {
        "unique_reference_route_signatures": len(heldout_route_signatures),
        "exact_route_matches_removed": skipped_route_holdout,
        "chemical_duplicates_removed": exact_chemical_duplicates_removed,
        "mapped_smiles_variants_collapsed": mapped_smiles_variants_collapsed,
        "duplicate_routes_removed": exact_chemical_duplicates_removed + mapped_smiles_variants_collapsed,
    }
    if config.holdout_mode == "reaction":
        postprocessing["reaction_overlap"] = {
            "unique_reference_reaction_signatures": len(heldout_reaction_signatures),
            "routes_with_overlapping_reactions": reaction_excision_source_routes,
            "fragments_kept_after_excision": reaction_excision_fragments,
            "routes_fully_removed_after_excision": fully_removed_by_reaction_excision,
        }

    return TrainingSetBuildResult(
        release_name=config.release_name,
        records=records,
        summary={
            "input": {
                "all_routes": all_adaptation.raw_routes,
            },
            "adaptation": {
                "all_routes": all_adaptation.to_manifest_dict(),
                "reference_datasets": {
                    dataset: stats.to_manifest_dict() for dataset, stats in sorted((heldout_adaptation or {}).items())
                },
            },
            "postprocessing": postprocessing,
            "output": summarize_records(records),
        },
    )


def assign_train_val_splits(
    routes_by_signature: dict[str, Route],
    val_fraction: float,
    seed: int,
) -> dict[str, SplitName]:
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    groups: dict[tuple[int, bool], list[str]] = defaultdict(list)
    for signature, route in routes_by_signature.items():
        key = (
            route.length,
            route.has_convergent_reaction,
        )
        groups[key].append(signature)

    split_by_signature: dict[str, SplitName] = {}
    rng = random.Random(seed)
    for key in sorted(groups):
        signatures = sorted(groups[key])
        rng.shuffle(signatures)
        n_val = _validation_count(len(signatures), val_fraction)
        val_signatures = set(signatures[:n_val])
        for signature in signatures:
            split_by_signature[signature] = "val" if signature in val_signatures else "train"

    return split_by_signature


def _validation_count(n_items: int, val_fraction: float) -> int:
    if n_items <= 1:
        return 0
    return min(n_items - 1, max(1, round(n_items * val_fraction)))


def summarize_records(records: Sequence[TrainingRouteRecord]) -> dict[str, Any]:
    training_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "val"]

    return {
        "all_records": split_counts(
            total=len(records),
            train=len(training_records),
            val=len(validation_records),
        ),
    }


def split_counts(total: int, train: int, val: int) -> dict[str, int]:
    return {
        "total": total,
        "training_split": train,
        "validation_split": val,
    }


def write_training_release(
    result: TrainingSetBuildResult,
    output_dir: Path,
    source_paths: Sequence[Path],
    config: TrainingSetBuildConfig,
    source_root: Path | None = None,
) -> dict[str, Any]:
    release_dir = output_dir / result.release_name
    release_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing training release to %s", release_dir)

    files = {
        "all": release_dir / "all.jsonl.gz",
        "train": release_dir / "train.jsonl.gz",
        "val": release_dir / "val.jsonl.gz",
    }
    grouped_records = {
        "all": result.records,
        "train": [record for record in result.records if record.split == "train"],
        "val": [record for record in result.records if record.split == "val"],
    }
    for name, records in grouped_records.items():
        logger.info("  writing %s (%s records)", files[name].name, f"{len(records):,}")
        save_jsonl_gz((record.to_json_dict() for record in records), files[name])

    manifest = build_training_manifest(
        release_name=result.release_name,
        files=files,
        source_paths=source_paths,
        source_root=source_root,
        config=config,
        summary=result.summary,
    )
    manifest_path = release_dir / "manifest.json"
    logger.info("  writing %s", manifest_path.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_training_manifest(
    release_name: str,
    files: dict[str, Path],
    source_paths: Sequence[Path],
    source_root: Path | None,
    config: TrainingSetBuildConfig,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    root_dir = source_root if source_root is not None else Path.cwd()
    manifest = create_manifest(
        action=TRAINING_RELEASE_ACTION,
        sources=list(source_paths),
        outputs=[(name, path, None, ContentType.UNKNOWN) for name, path in sorted(files.items())],
        root_dir=root_dir,
        parameters=config.to_manifest_dict(),
        summary=dict(summary),
        release_name=release_name,
        keyed_output_files=True,
    )
    return manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
