from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from retrocast.adapters.base_adapter import RawRouteEntry
from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.adapters.paroutes_diagnostics import (
    ConditionSlotParseStatistics,
    PatentIdParseStatistics,
    collect_raw_paroutes_route_diagnostics,
    log_patent_id_parse_statistics,
)
from retrocast.curation.filtering import deduplicate_routes, excise_reactions_from_route
from retrocast.curation.training.records import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    ConditionIdentity,
    PreparedTrainingRoute,
    RawRouteSource,
    SplitName,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
    TrainingSetBuildResult,
)
from retrocast.exceptions import AdapterError, ChemError, TrainingReleaseError
from retrocast.io import ContentType, create_manifest, save_jsonl_gz
from retrocast.models.chem import ReactionSignature, Route, TargetInput
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)
TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"


@dataclass(frozen=True, slots=True)
class ReleaseStep:
    mapped_smiles: str | None
    condition_slot: str | None
    condition_slot_smiles: tuple[SmilesStr, ...]
    alternative_mapped_smiles: tuple[str, ...]

    @property
    def condition_identity(self) -> ConditionIdentity:
        return self.condition_slot_smiles or self.condition_slot


def stable_raw_route_hash(raw_route: dict[str, Any]) -> str:
    payload = json.dumps(raw_route, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def iter_training_raw_route_entries(
    raw_routes: Sequence[dict[str, Any]],
    *,
    dataset: str,
    id_width: int,
) -> Sequence[RawRouteEntry]:
    entries: list[RawRouteEntry] = []
    for idx, raw_route in enumerate(raw_routes):
        raw_smiles = raw_route.get("smiles") if isinstance(raw_route, Mapping) else None
        entries.append(
            RawRouteEntry(
                payload=raw_route,
                source_key=dataset,
                source_row_index=idx,
                target_hint_id=f"{dataset}-{idx + 1:0{id_width}d}",
                target_hint_smiles=raw_smiles if isinstance(raw_smiles, str) and raw_smiles else None,
                source_order=idx + 1,
            )
        )
    return entries


def adapt_training_routes(
    raw_routes: Sequence[dict[str, Any]],
    dataset: str,
    id_width: int,
    collect_reactions: bool,
    show_progress: bool = True,
) -> tuple[list[AdaptedTrainingRoute], AdaptationStatistics]:
    adapter = PaRoutesAdapter()
    parse_stats = ConditionSlotParseStatistics()
    patent_id_stats = PatentIdParseStatistics()
    adapted_routes: list[AdaptedTrainingRoute] = []
    failures_by_code: dict[str, int] = defaultdict(int)
    reaction_hash_signature_pairs: set[tuple[str, ReactionSignature]] = set()
    skipped_adaptation = 0
    skipped_without_error_code = 0
    raw_entries = iter_training_raw_route_entries(raw_routes, dataset=dataset, id_width=id_width)
    entries = (
        tqdm(raw_entries, desc=f"adapt {dataset}", unit="route", dynamic_ncols=True, leave=False)
        if show_progress
        else raw_entries
    )
    for entry in entries:
        if entry.target_hint_smiles is None or entry.target_hint_id is None:
            skipped_adaptation += 1
            failures_by_code["adapter.schema_invalid"] += 1
            continue

        target = TargetInput(id=entry.target_hint_id, smiles=entry.target_hint_smiles)
        raw_route = entry.payload
        assert isinstance(raw_route, dict), "training route entries must preserve raw dict payloads"
        collect_raw_paroutes_route_diagnostics(
            raw_route,
            patent_id_parse_statistics=patent_id_stats,
            condition_slot_parse_statistics=parse_stats,
        )
        try:
            route = adapter.cast(
                raw_route,
                expected_target=target,
            )
        except (AdapterError, ChemError) as exc:
            skipped_adaptation += 1
            failures_by_code[exc.code] += 1
            continue

        patent_id = route.metadata.get("patent_id") if isinstance(route.metadata.get("patent_id"), str) else None
        collect_reaction_hash_sanity(route, raw_route, reaction_hash_signature_pairs)
        adapted_routes.append(
            AdaptedTrainingRoute(
                route=route,
                structural_signature=route.get_structural_signature(),
                reaction_signatures=route.get_reaction_signatures() if collect_reactions else set(),
                source=RawRouteSource(
                    dataset=dataset,
                    raw_index=entry.source_row_index or 0,
                    raw_route_hash=stable_raw_route_hash(raw_route),
                    patent_id=patent_id,
                ),
            )
        )
    assert_paroutes_reaction_hash_matches_retrocast_signature(reaction_hash_signature_pairs)
    log_patent_id_parse_statistics(patent_id_stats, logger_name=__name__)
    if parse_stats.malformed_rsmi_count or parse_stats.uncanonicalizable_token_count:
        logger.info(
            "PaRoutes condition-slot parsing for %s skipped %s malformed rsmi slots and %s "
            "uncanonicalizable tokens during best-effort metadata extraction. this is non-fatal "
            "and does not affect adapted route counts.",
            dataset,
            parse_stats.malformed_rsmi_count,
            parse_stats.uncanonicalizable_token_count,
        )
        if top_examples := parse_stats.top_uncanonicalizable_tokens:
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
        non_fatal_condition_slot_parse=parse_stats,
        patent_id_parse=patent_id_stats,
    )


@dataclass
class TrainingRouteReleaseBuilder:
    all_routes: Sequence[AdaptedTrainingRoute]
    all_adaptation: AdaptationStatistics
    holdout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]]
    holdout_adaptation: Mapping[str, AdaptationStatistics]
    config: TrainingSetBuildConfig
    holdout_route_signatures: set[str] = dataclass_field(default_factory=set, init=False)
    holdout_reaction_signatures: set[ReactionSignature] = dataclass_field(default_factory=set, init=False)
    skipped_route_holdout: int = dataclass_field(default=0, init=False)
    reaction_excision_source_routes: int = dataclass_field(default=0, init=False)
    reaction_excision_fragments: int = dataclass_field(default=0, init=False)
    fully_removed_by_reaction_excision: int = dataclass_field(default=0, init=False)
    exact_chemical_duplicates_removed: int = dataclass_field(default=0, init=False)
    mapped_smiles_variants_collapsed: int = dataclass_field(default=0, init=False)
    _started: bool = dataclass_field(default=False, init=False)

    def build(self) -> TrainingSetBuildResult:
        self.assert_not_started()
        self._started = True
        self.collect_holdout_signatures()
        candidates = self.apply_holdout()
        chemically_unique_routes, self.exact_chemical_duplicates_removed = merge_exact_chemical_duplicates(candidates)
        prepared_routes, self.mapped_smiles_variants_collapsed = merge_transform_equivalent_routes(
            chemically_unique_routes
        )
        records = self.materialize_records(prepared_routes)
        return TrainingSetBuildResult(
            release_name=self.config.release_name,
            records=records,
            summary={
                **self.summary(),
                "output": summarize_records(records),
            },
        )

    def assert_not_started(self) -> None:
        if (
            self._started
            or self.holdout_route_signatures
            or self.holdout_reaction_signatures
            or self.skipped_route_holdout
            or self.reaction_excision_source_routes
            or self.reaction_excision_fragments
            or self.fully_removed_by_reaction_excision
            or self.exact_chemical_duplicates_removed
            or self.mapped_smiles_variants_collapsed
        ):
            raise RuntimeError("TrainingRouteReleaseBuilder instances are single-use")

    def collect_holdout_signatures(self) -> None:
        for routes in self.holdout_routes.values():
            for route in routes:
                self.holdout_route_signatures.add(route.structural_signature)
                if self.config.holdout_mode == "reaction":
                    self.holdout_reaction_signatures.update(route.reaction_signatures)

    def apply_holdout(self) -> list[AdaptedTrainingRoute]:
        candidates: list[AdaptedTrainingRoute] = []
        for route in self.all_routes:
            if route.structural_signature in self.holdout_route_signatures:
                self.skipped_route_holdout += 1
                continue
            if self.config.holdout_mode == "reaction":
                candidates.extend(self.excise_holdout_reactions(route))
            else:
                candidates.append(route)
        return candidates

    def excise_holdout_reactions(self, route: AdaptedTrainingRoute) -> list[AdaptedTrainingRoute]:
        overlapping_signatures = route.reaction_signatures & self.holdout_reaction_signatures
        if not overlapping_signatures:
            return [route]

        fragments = deduplicate_routes(excise_reactions_from_route(route.route, overlapping_signatures))
        self.reaction_excision_source_routes += 1
        self.reaction_excision_fragments += len(fragments)
        if not fragments:
            self.fully_removed_by_reaction_excision += 1
        return [
            AdaptedTrainingRoute(
                route=fragment,
                structural_signature=fragment.get_structural_signature(),
                reaction_signatures=fragment.get_reaction_signatures(),
                source=route.source,
            )
            for fragment in fragments
        ]

    def materialize_records(self, prepared_routes: Sequence[PreparedTrainingRoute]) -> list[TrainingRouteRecord]:
        routes_by_signature: dict[str, Route] = {}
        for route in prepared_routes:
            routes_by_signature.setdefault(route.structural_signature, route.route)

        split_by_signature = assign_train_val_splits(
            routes_by_signature,
            val_fraction=self.config.val_fraction,
            seed=self.config.seed,
        )

        records: list[TrainingRouteRecord] = []
        for ordinal, prepared_route in enumerate(prepared_routes, start=1):
            route = prepared_route.route.model_copy(deep=True)
            route.metadata = build_release_route_metadata(route, prepared_route.sources)
            records.append(
                TrainingRouteRecord(
                    id=f"{self.config.route_prefix}-{self.config.release_name}-{ordinal:06d}",
                    split=split_by_signature[prepared_route.structural_signature],
                    route=route,
                    sources=prepared_route.sources,
                )
            )

        assert_no_cross_split_route_signature_overlap(records)
        return records

    def summary(self) -> dict[str, Any]:
        postprocessing: dict[str, Any] = {
            "unique_reference_route_signatures": len(self.holdout_route_signatures),
            "exact_route_matches_removed": self.skipped_route_holdout,
            "chemical_duplicates_removed": self.exact_chemical_duplicates_removed,
            "mapped_smiles_variants_collapsed": self.mapped_smiles_variants_collapsed,
            "duplicate_routes_removed": self.exact_chemical_duplicates_removed + self.mapped_smiles_variants_collapsed,
        }
        if self.config.holdout_mode == "reaction":
            postprocessing["reaction_overlap"] = {
                "unique_reference_reaction_signatures": len(self.holdout_reaction_signatures),
                "routes_with_overlapping_reactions": self.reaction_excision_source_routes,
                "fragments_kept_after_excision": self.reaction_excision_fragments,
                "routes_fully_removed_after_excision": self.fully_removed_by_reaction_excision,
            }
        return {
            "input": {"all_routes": self.all_adaptation.raw_routes},
            "adaptation": {
                "all_routes": self.all_adaptation.to_manifest_dict(),
                "reference_datasets": {
                    dataset: stats.to_manifest_dict() for dataset, stats in sorted(self.holdout_adaptation.items())
                },
            },
            "postprocessing": postprocessing,
        }


def collect_reaction_hash_sanity(
    route: Route,
    raw_route: Mapping[str, Any],
    reaction_hash_signature_pairs: set[tuple[str, ReactionSignature]],
) -> None:
    reaction_hash_by_source_id = extract_paroutes_reaction_hash_by_source_id(raw_route)
    for route_reaction in route.iter_reactions():
        step = route_reaction.step
        source_id = step.metadata.get("source_id")
        reaction_hash = reaction_hash_by_source_id.get(source_id) if isinstance(source_id, str) else None
        if reaction_hash is None:
            continue
        reaction_hash_signature_pairs.add((reaction_hash, route_reaction.signature))


def assert_paroutes_reaction_hash_matches_retrocast_signature(
    reaction_hash_signature_pairs: set[tuple[str, ReactionSignature]],
) -> None:
    n_pairs = len(reaction_hash_signature_pairs)
    n_hashes = len({reaction_hash for reaction_hash, _ in reaction_hash_signature_pairs})
    n_signatures = len({signature for _, signature in reaction_hash_signature_pairs})
    if n_pairs != n_hashes or n_pairs != n_signatures:
        raise TrainingReleaseError(
            "PaRoutes reaction_hash is not equivalent to RetroCast reaction signatures: "
            f"{n_pairs} unique pairs, {n_hashes} unique hashes, {n_signatures} unique signatures",
            code="workflow.paroutes_reaction_hash_mismatch",
            context={"unique_pairs": n_pairs, "unique_hashes": n_hashes, "unique_signatures": n_signatures},
        )


def extract_paroutes_reaction_hash_by_source_id(raw_route: Mapping[str, Any]) -> dict[str, str]:
    """Return raw PaRoutes reaction hashes keyed by reaction source id.

    PaRoutes reaction nodes carry side metadata like:

        {"metadata": {"ID": "US20150051201A1;outer", "reaction_hash": "outer-hash"}}

    In the source data, `reaction_hash` is PaRoutes' reaction identity: the reaction SMILES rebuilt from molecule InChIKeys. RetroCast's equivalent is `ReactionSignature`, the `(frozenset(reactant.inchikey), product.inchikey)` tuple produced by `Route.get_reaction_signatures()` and reconstructed from `Route.iter_reactions()` in the adaptation sanity check.

    This helper exists only for that check. If each PaRoutes hash maps one-to-one with one RetroCast signature, the release pipeline drops the raw hash instead of carrying a PaRoutes-specific identity sidecar through deduplication.
    """
    reaction_hash_by_source_id: dict[str, str] = {}
    stack: list[Mapping[str, Any]] = [raw_route]
    while stack:
        node = stack.pop()
        metadata = node.get("metadata")
        if isinstance(metadata, Mapping):
            source_id = metadata.get("ID")
            reaction_hash = metadata.get("reaction_hash")
            if isinstance(source_id, str) and isinstance(reaction_hash, str) and reaction_hash:
                reaction_hash_by_source_id[source_id] = reaction_hash

        if isinstance(children := node.get("children"), list):
            stack.extend(child for child in children if isinstance(child, Mapping))
    return reaction_hash_by_source_id


def release_steps(route: Route) -> tuple[ReleaseStep, ...]:
    return tuple(release_step(step) for step in route.iter_steps())


def release_step(step: Any) -> ReleaseStep:
    metadata = step.metadata
    condition_slot = metadata.get("condition_slot")
    raw_smiles = metadata.get("condition_slot_smiles")
    raw_alternatives = metadata.get("alternative_mapped_smiles")
    return ReleaseStep(
        mapped_smiles=step.mapped_smiles,
        condition_slot=condition_slot if isinstance(condition_slot, str) and condition_slot else None,
        condition_slot_smiles=tuple(raw_smiles)
        if isinstance(raw_smiles, list) and all(isinstance(value, str) for value in raw_smiles)
        else (),
        alternative_mapped_smiles=tuple(sorted(value for value in (raw_alternatives or []) if isinstance(value, str))),
    )


def get_step_condition_slot_smiles(step: Any) -> tuple[SmilesStr, ...]:
    return release_step(step).condition_slot_smiles


def build_release_route_metadata(route: Route, sources: Sequence[RawRouteSource]) -> dict[str, Any]:
    patent_ids = sorted({source.patent_id for source in sources if source.patent_id is not None})
    return {
        **{key: value for key, value in route.metadata.items() if key not in ("patent_id", "source_patent_ids")},
        **({"source_patent_ids": patent_ids} if patent_ids else {}),
    }


def merge_alternative_mapped_smiles(canonical_route: Route, routes: Sequence[Route]) -> None:
    canonical_steps = canonical_route.iter_steps()
    route_steps = [route.iter_steps() for route in routes]
    if any(len(steps) != len(canonical_steps) for steps in route_steps):
        raise TrainingReleaseError(
            "cannot merge mapped smiles for routes with different step counts",
            code="workflow.route_mapped_smiles_merge_step_count_mismatch",
            context={
                "canonical_step_count": len(canonical_steps),
                "candidate_step_counts": [len(steps) for steps in route_steps],
            },
        )
    for step_index, canonical_step in enumerate(canonical_steps):
        variants = {
            value
            for step in (steps[step_index] for steps in [canonical_steps, *route_steps])
            for value in (*release_step(step).alternative_mapped_smiles, step.mapped_smiles)
            if value is not None
        }
        alternatives = sorted(
            mapped_smiles for mapped_smiles in variants if mapped_smiles != canonical_step.mapped_smiles
        )
        canonical_step.metadata = canonical_step.metadata.copy()
        if alternatives:
            canonical_step.metadata["alternative_mapped_smiles"] = alternatives
        else:
            canonical_step.metadata.pop("alternative_mapped_smiles", None)


def merge_exact_chemical_duplicates(routes: Sequence[AdaptedTrainingRoute]) -> tuple[list[PreparedTrainingRoute], int]:
    routes_by_signature: dict[str, PreparedTrainingRoute] = {}
    duplicates_removed = 0

    for adapted_route in routes:
        exact_signature = adapted_route.route.get_annotated_signature(include_mapped_smiles=True)
        existing_route = routes_by_signature.get(exact_signature)
        if existing_route is None:
            routes_by_signature[exact_signature] = PreparedTrainingRoute(
                route=adapted_route.route.model_copy(deep=True),
                structural_signature=adapted_route.structural_signature,
                sources=[adapted_route.source],
            )
            continue

        existing_route.sources.append(adapted_route.source)
        duplicates_removed += 1

    return list(routes_by_signature.values()), duplicates_removed


def merge_transform_equivalent_routes(
    routes: Sequence[PreparedTrainingRoute],
) -> tuple[list[PreparedTrainingRoute], int]:
    grouped_routes: dict[tuple[str, tuple[ConditionIdentity, ...]], list[PreparedTrainingRoute]] = defaultdict(list)
    for route in routes:
        step_keys = tuple(step.condition_identity for step in release_steps(route.route))
        grouped_routes[(route.structural_signature, tuple(step_keys))].append(route)

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
            profile = tuple(step.mapped_smiles for step in release_steps(route.route))
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
        merged_routes.append(
            PreparedTrainingRoute(
                route=merged_route,
                structural_signature=canonical_source.structural_signature,
                sources=merged_sources,
            )
        )

    return merged_routes, duplicates_removed


def assign_train_val_splits(
    routes_by_signature: dict[str, Route],
    val_fraction: float,
    seed: int,
) -> dict[str, SplitName]:
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    groups: dict[tuple[int, bool], list[str]] = defaultdict(list)
    for signature, route in routes_by_signature.items():
        groups[(route.length, route.has_convergent_reaction)].append(signature)

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
    return {
        "all_records": split_counts(
            total=len(records),
            training=sum(record.split == "training" for record in records),
            validation=sum(record.split == "validation" for record in records),
        ),
    }


def split_counts(total: int, training: int, validation: int) -> dict[str, int]:
    return {"total": total, "training": training, "validation": validation}


def assert_no_cross_split_route_signature_overlap(records: Sequence[TrainingRouteRecord]) -> None:
    split_by_signature: dict[str, SplitName] = {}
    overlapping_signatures: set[str] = set()
    for record in records:
        existing_split = split_by_signature.setdefault(record.route_signature, record.split)
        if existing_split != record.split:
            overlapping_signatures.add(record.route_signature)

    if overlapping_signatures:
        raise TrainingReleaseError(
            "training route release has route_signature overlap across training and validation splits: "
            f"{len(overlapping_signatures)} overlapping signatures",
            code="workflow.route_split_signature_overlap",
            context={"overlapping_signature_count": len(overlapping_signatures)},
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
