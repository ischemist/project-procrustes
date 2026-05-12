from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from tqdm.auto import tqdm

from retrocast.adapters import adapt_single_route_with_diagnostics, get_adapter
from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.curation.filtering import deduplicate_routes, excise_reactions_from_route
from retrocast.curation.training.records import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    ConditionIdentity,
    NonFatalConditionSlotParseStatistics,
    PreparedTrainingRoute,
    PreparedTrainingSetBuildResult,
    RawRouteSource,
    SplitName,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
    TrainingSetBuildResult,
    TransformDedupKey,
    TransformDedupStepKey,
)
from retrocast.io import ContentType, create_manifest, save_jsonl_gz
from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route, TargetInput
from retrocast.typing import SmilesStr

T = TypeVar("T")

logger = logging.getLogger(__name__)
TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"


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


def iter_route_reactions(route: Route) -> list[tuple[Molecule, ReactionStep]]:
    """Return product/step pairs in deterministic root-first depth-first order."""
    reactions: list[tuple[Molecule, ReactionStep]] = []

    def _visit(node: Molecule) -> None:
        if node.synthesis_step is None:
            return
        reactions.append((node, node.synthesis_step))
        for reactant in node.synthesis_step.reactants:
            _visit(reactant)

    _visit(route.target)
    return reactions


def get_training_route_record_id(config: TrainingSetBuildConfig, ordinal: int) -> str:
    return f"{config.route_prefix}-{config.release_name}-{ordinal:06d}"


def get_step_condition_identity(step: ReactionStep) -> ConditionIdentity:
    """Condition identity used for transform-level deduplication."""
    condition_slot_smiles = step.metadata.get("condition_slot_smiles")
    if isinstance(condition_slot_smiles, list) and all(isinstance(value, str) for value in condition_slot_smiles):
        return tuple(condition_slot_smiles)

    condition_slot = step.metadata.get("condition_slot")
    if isinstance(condition_slot, str) and condition_slot:
        return condition_slot

    return None


def get_step_condition_slot_smiles(step: ReactionStep) -> tuple[SmilesStr, ...]:
    """Structured molecules parsed from the condition slot, when available."""
    condition_slot_smiles = step.metadata.get("condition_slot_smiles")
    if isinstance(condition_slot_smiles, list) and all(isinstance(value, str) for value in condition_slot_smiles):
        return tuple(condition_slot_smiles)
    return ()


def get_transform_dedup_key(route: Route, *, transform_ids_by_source_id: Mapping[str, str]) -> TransformDedupKey:
    """Identity for collapsing mapped-smiles drift while preserving condition variants."""
    step_keys: list[TransformDedupStepKey] = []
    for step in iter_route_steps(route):
        source_id = step.metadata.get("source_id")
        transform_id = transform_ids_by_source_id.get(source_id) if isinstance(source_id, str) else None
        step_keys.append((transform_id, get_step_condition_identity(step)))
    return route.get_structural_signature(), tuple(step_keys)


def sync_route_source_metadata(route: Route, sources: Sequence[RawRouteSource]) -> None:
    """Keep released route metadata honest after provenance-preserving merges."""
    route.metadata = route.metadata.copy()
    route.metadata.pop("patent_id", None)

    source_patent_ids = sorted({source.patent_id for source in sources if source.patent_id is not None})
    if source_patent_ids:
        route.metadata["source_patent_ids"] = source_patent_ids
    else:
        route.metadata.pop("source_patent_ids", None)


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
        exact_signature = adapted_route.route.get_annotated_signature(include_mapped_smiles=True)
        existing_route = routes_by_signature.get(exact_signature)
        if existing_route is None:
            cloned_route = adapted_route.route.model_copy(deep=True)
            sync_route_source_metadata(cloned_route, [adapted_route.source])
            routes_by_signature[exact_signature] = PreparedTrainingRoute(
                route=cloned_route,
                structural_signature=adapted_route.structural_signature,
                sources=[adapted_route.source],
                transform_ids_by_source_id=dict(adapted_route.transform_ids_by_source_id),
            )
            continue

        existing_route.sources.append(adapted_route.source)
        sync_route_source_metadata(existing_route.route, existing_route.sources)
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
            profile = tuple(step.mapped_smiles for step in iter_route_steps(route.route))
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

        merged_route = canonical_source.route.model_copy(deep=True)
        merge_alternative_mapped_smiles(merged_route, [route.route for route in group])
        merged_sources = [source for route in group for source in route.sources]
        sync_route_source_metadata(merged_route, merged_sources)
        merged_routes.append(
            PreparedTrainingRoute(
                route=merged_route,
                structural_signature=canonical_source.structural_signature,
                sources=merged_sources,
                transform_ids_by_source_id=dict(canonical_source.transform_ids_by_source_id),
            )
        )

    return merged_routes, duplicates_removed


def prepare_training_routes_from_adapted(
    all_routes: Sequence[AdaptedTrainingRoute],
    all_adaptation: AdaptationStatistics,
    heldout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
    heldout_adaptation: Mapping[str, AdaptationStatistics] | None,
    config: TrainingSetBuildConfig,
) -> PreparedTrainingSetBuildResult:
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
    prepared_routes, mapped_smiles_variants_collapsed = merge_transform_equivalent_routes(chemically_unique_routes)

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

    return PreparedTrainingSetBuildResult(
        release_name=config.release_name,
        prepared_routes=prepared_routes,
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
        },
    )


def materialize_training_route_records(
    prepared_routes: Sequence[PreparedTrainingRoute],
    config: TrainingSetBuildConfig,
) -> list[TrainingRouteRecord]:
    routes_by_signature: dict[str, Route] = {}
    for prepared_route in prepared_routes:
        routes_by_signature.setdefault(prepared_route.structural_signature, prepared_route.route)

    split_by_signature = assign_train_val_splits(
        routes_by_signature,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    records: list[TrainingRouteRecord] = []
    for ordinal, prepared_route in enumerate(prepared_routes, start=1):
        route = prepared_route.route
        records.append(
            TrainingRouteRecord(
                id=get_training_route_record_id(config, ordinal),
                split=split_by_signature[prepared_route.structural_signature],
                route_signature=prepared_route.structural_signature,
                content_hash=route.get_content_hash(),
                route=route,
                sources=prepared_route.sources,
            )
        )

    assert_no_cross_split_route_signature_overlap(records)
    return records


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
    adapter = get_adapter("paroutes")
    if isinstance(adapter, PaRoutesAdapter):
        adapter.reset_condition_slot_parse_statistics()

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
    non_fatal_condition_slot_parse: NonFatalConditionSlotParseStatistics | None = None
    if isinstance(adapter, PaRoutesAdapter):
        condition_slot_stats = adapter.get_condition_slot_parse_statistics()
        malformed_rsmi_count = condition_slot_stats["malformed_rsmi_count"]
        uncanonicalizable_token_count = condition_slot_stats["uncanonicalizable_token_count"]
        distinct_uncanonicalizable_token_count = condition_slot_stats["distinct_uncanonicalizable_token_count"]
        non_fatal_condition_slot_parse = NonFatalConditionSlotParseStatistics(
            malformed_rsmi_count=malformed_rsmi_count,
            uncanonicalizable_token_count=uncanonicalizable_token_count,
            distinct_uncanonicalizable_token_count=distinct_uncanonicalizable_token_count,
        )
        if malformed_rsmi_count or uncanonicalizable_token_count:
            logger.info(
                "PaRoutes condition-slot parsing for %s skipped %s malformed rsmi slots and %s "
                "uncanonicalizable tokens during best-effort metadata extraction. this is non-fatal "
                "and does not affect adapted route counts.",
                dataset,
                malformed_rsmi_count,
                uncanonicalizable_token_count,
            )
            top_examples = condition_slot_stats["top_uncanonicalizable_tokens"]
            if top_examples:
                logger.info(
                    "  top uncanonicalizable condition-slot tokens for %s: %s",
                    dataset,
                    {token: count for token, count in top_examples},
                )
    return adapted_routes, AdaptationStatistics(
        raw_routes=len(raw_routes),
        adapted_routes=len(adapted_routes),
        skipped_routes=skipped_adaptation,
        skipped_without_error_code=skipped_without_error_code,
        failures_by_code=dict(failures_by_code),
        non_fatal_condition_slot_parse=non_fatal_condition_slot_parse,
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
    prepared_result = prepare_training_routes_from_adapted(
        all_routes=all_routes,
        all_adaptation=all_adaptation,
        heldout_routes=heldout_routes,
        heldout_adaptation=heldout_adaptation,
        config=config,
    )
    records = materialize_training_route_records(prepared_result.prepared_routes, config)
    return TrainingSetBuildResult(
        release_name=prepared_result.release_name,
        records=records,
        summary={
            **prepared_result.summary,
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
            split_by_signature[signature] = "validation" if signature in val_signatures else "training"

    return split_by_signature


def _validation_count(n_items: int, val_fraction: float) -> int:
    if n_items <= 1:
        return 0
    return min(n_items - 1, max(1, round(n_items * val_fraction)))


def summarize_records(records: Sequence[TrainingRouteRecord]) -> dict[str, Any]:
    training_records = [record for record in records if record.split == "training"]
    validation_records = [record for record in records if record.split == "validation"]

    return {
        "all_records": split_counts(
            total=len(records),
            training=len(training_records),
            validation=len(validation_records),
        ),
    }


def split_counts(total: int, training: int, validation: int) -> dict[str, int]:
    return {
        "total": total,
        "training": training,
        "validation": validation,
    }


def assert_no_cross_split_route_signature_overlap(records: Sequence[TrainingRouteRecord]) -> None:
    split_by_signature: dict[str, SplitName] = {}
    overlapping_signatures: set[str] = set()
    for record in records:
        existing_split = split_by_signature.setdefault(record.route_signature, record.split)
        if existing_split != record.split:
            overlapping_signatures.add(record.route_signature)

    if overlapping_signatures:
        raise ValueError(
            "training route release has route_signature overlap across training and validation splits: "
            f"{len(overlapping_signatures)} overlapping signatures"
        )


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
        "training": release_dir / "training.jsonl.gz",
        "validation": release_dir / "validation.jsonl.gz",
    }
    grouped_records = {
        "all": result.records,
        "training": [record for record in result.records if record.split == "training"],
        "validation": [record for record in result.records if record.split == "validation"],
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
        action=TRAINING_RELEASE_ACTION,
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
    action: str,
) -> dict[str, Any]:
    root_dir = source_root if source_root is not None else Path.cwd()
    manifest = create_manifest(
        action=action,
        sources=list(source_paths),
        outputs=[(name, path, None, ContentType.UNKNOWN) for name, path in sorted(files.items())],
        root_dir=root_dir,
        parameters=config.to_manifest_dict(),
        summary=dict(summary),
        release_name=release_name,
        keyed_output_files=True,
    )
    return manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
