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
from retrocast.models.chem import ReactionSignature, Route, TargetInput

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["train", "val"]
T = TypeVar("T")

logger = logging.getLogger(__name__)
TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"


@dataclass(frozen=True)
class RawRouteSource:
    dataset: str
    raw_index: int
    raw_route_hash: str


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
                "deduplication_key": "route.get_signature()",
                "route_holdout": "exclude full n1 union n5 by route.get_signature()",
                "reaction_holdout": "excise reactions present in n1 union n5 and keep surviving sub-routes",
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
    route_signature: str
    reaction_signatures: set[ReactionSignature]
    source: RawRouteSource


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
        adapted_routes.append(
            AdaptedTrainingRoute(
                route=route,
                route_signature=route.get_signature(),
                reaction_signatures=route.get_reaction_signatures() if collect_reactions else set(),
                source=RawRouteSource(
                    dataset=dataset,
                    raw_index=idx,
                    raw_route_hash=stable_raw_route_hash(raw_route),
                ),
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
            route_signatures.add(route.route_signature)
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
            route_signature=route.get_signature(),
            reaction_signatures=route.get_reaction_signatures(),
            source=adapted_route.source,
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

    routes_by_signature: dict[str, Route] = {}
    sources_by_signature: dict[str, list[RawRouteSource]] = defaultdict(list)
    skipped_route_holdout = 0
    duplicate_routes = 0
    reaction_excision_source_routes = 0
    reaction_excision_fragments = 0
    fully_removed_by_reaction_excision = 0

    for adapted_route in all_routes:
        if adapted_route.route_signature in heldout_route_signatures:
            skipped_route_holdout += 1
            continue

        candidate_routes = [adapted_route]
        if config.holdout_mode == "reaction":
            candidate_routes, was_excised = excise_heldout_reactions(adapted_route, heldout_reaction_signatures)
            if was_excised:
                reaction_excision_source_routes += 1
                reaction_excision_fragments += len(candidate_routes)
                if not candidate_routes:
                    fully_removed_by_reaction_excision += 1

        for candidate_route in candidate_routes:
            if candidate_route.route_signature in routes_by_signature:
                duplicate_routes += 1
            else:
                routes_by_signature[candidate_route.route_signature] = candidate_route.route

            sources_by_signature[candidate_route.route_signature].append(candidate_route.source)

    split_by_signature = assign_train_val_splits(
        routes_by_signature,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    records: list[TrainingRouteRecord] = []
    for ordinal, route_signature in enumerate(sorted(routes_by_signature), start=1):
        route = routes_by_signature[route_signature]
        records.append(
            TrainingRouteRecord(
                id=f"{config.route_prefix}-{config.release_name}-{ordinal:06d}",
                split=split_by_signature[route_signature],
                route_signature=route_signature,
                content_hash=route.get_content_hash(),
                route=route,
                sources=sources_by_signature[route_signature],
            )
        )

    return TrainingSetBuildResult(
        release_name=config.release_name,
        records=records,
        summary={
            "input": {
                "all_routes": all_adaptation.raw_routes,
            },
            "adaptation": {
                "all": all_adaptation.to_manifest_dict(),
                "heldout": {
                    dataset: stats.to_manifest_dict() for dataset, stats in sorted((heldout_adaptation or {}).items())
                },
            },
            "holdout": {
                "route_signatures": len(heldout_route_signatures),
                "reaction_signatures": len(heldout_reaction_signatures),
                "excluded_routes": {"route": skipped_route_holdout},
                "reaction_excision": {
                    "source_routes_with_overlap": reaction_excision_source_routes,
                    "surviving_fragments": reaction_excision_fragments,
                    "fully_removed_source_routes": fully_removed_by_reaction_excision,
                },
            },
            "deduplication": {
                "duplicate_routes_removed": duplicate_routes,
            },
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
