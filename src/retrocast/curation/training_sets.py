from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from tqdm.auto import tqdm

from retrocast import __version__
from retrocast.adapters import adapt_single_route_with_diagnostics
from retrocast.chem import InchiKeyLevel
from retrocast.curation.filtering import deduplicate_routes, excise_reactions_from_route
from retrocast.io import save_jsonl_gz
from retrocast.io.provenance import calculate_file_hash
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.chem import ReactionSignature, Route, TargetInput
from retrocast.typing import InchiKeyStr

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["train", "val"]
T = TypeVar("T")

logger = logging.getLogger(__name__)


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
    buyables_solved: bool
    n_leaves: int
    n_buyable_leaves: int
    missing_buyable_leaf_inchikeys: list[str]
    route: Route
    sources: list[RawRouteSource] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "route_signature": self.route_signature,
            "content_hash": self.content_hash,
            "buyables_solved": self.buyables_solved,
            "n_leaves": self.n_leaves,
            "n_buyable_leaves": self.n_buyable_leaves,
            "missing_buyable_leaf_inchikeys": self.missing_buyable_leaf_inchikeys,
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
    stock_match_level: InchiKeyLevel = InchiKeyLevel.FULL
    show_progress: bool = True

    @property
    def release_name(self) -> str:
        return f"{self.holdout_mode}-heldout-n1-n5"


@dataclass
class TrainingSetBuildResult:
    release_name: str
    records: list[TrainingRouteRecord]
    summary: TrainingReleaseSummary


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


@dataclass(frozen=True)
class ReactionExcisionStatistics:
    source_routes_with_overlap: int
    surviving_fragments: int
    fully_removed_source_routes: int

    def to_manifest_dict(self) -> dict[str, int]:
        return {
            "source_routes_with_overlap": self.source_routes_with_overlap,
            "surviving_fragments": self.surviving_fragments,
            "fully_removed_source_routes": self.fully_removed_source_routes,
        }


@dataclass(frozen=True)
class SplitCounts:
    total: int
    training_split: int
    validation_split: int

    def to_manifest_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "training_split": self.training_split,
            "validation_split": self.validation_split,
        }


@dataclass(frozen=True)
class OutputSummary:
    all_records: SplitCounts
    stock_constrained_records: SplitCounts

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "all_records": self.all_records.to_manifest_dict(),
            "stock_constrained_records": self.stock_constrained_records.to_manifest_dict(),
        }


@dataclass(frozen=True)
class HoldoutSummary:
    route_signatures: int
    reaction_signatures: int
    exact_route_matches_excluded: int
    reaction_excision: ReactionExcisionStatistics

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "route_signatures": self.route_signatures,
            "reaction_signatures": self.reaction_signatures,
            "excluded_routes": {"route": self.exact_route_matches_excluded},
            "reaction_excision": self.reaction_excision.to_manifest_dict(),
        }


@dataclass(frozen=True)
class AdaptationSummary:
    all: AdaptationStatistics
    heldout: Mapping[str, AdaptationStatistics]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "all": self.all.to_manifest_dict(),
            "heldout": {dataset: stats.to_manifest_dict() for dataset, stats in sorted(self.heldout.items())},
        }


@dataclass(frozen=True)
class TrainingReleaseSummary:
    input_all_routes: int
    adaptation: AdaptationSummary
    holdout: HoldoutSummary
    duplicate_routes_removed: int
    output: OutputSummary

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "input": {
                "all_routes": self.input_all_routes,
            },
            "adaptation": self.adaptation.to_manifest_dict(),
            "holdout": self.holdout.to_manifest_dict(),
            "deduplication": {
                "duplicate_routes_removed": self.duplicate_routes_removed,
            },
            "output": self.output.to_manifest_dict(),
        }

    def to_log_lines(self) -> list[str]:
        lines = [
            (
                "Output: "
                f"{self.output.all_records.total} records "
                f"({self.output.all_records.training_split} train, {self.output.all_records.validation_split} validation); "
                "stock-constrained: "
                f"{self.output.stock_constrained_records.total} "
                f"({self.output.stock_constrained_records.training_split} train, "
                f"{self.output.stock_constrained_records.validation_split} validation)."
            ),
            (
                "Holdout: "
                f"{self.holdout.exact_route_matches_excluded} exact route matches excluded; "
                f"{self.holdout.reaction_excision.source_routes_with_overlap} overlapping routes excised into "
                f"{self.holdout.reaction_excision.surviving_fragments} surviving fragments "
                f"({self.holdout.reaction_excision.fully_removed_source_routes} fully removed)."
            ),
        ]
        if self.adaptation.all.failures_by_code:
            lines.append(f"Adaptation failures by code: {self.adaptation.all.failures_by_code}")
        return lines


@dataclass(frozen=True)
class TrainingReleaseParameters:
    holdout_mode: TrainingHoldoutMode
    val_fraction: float
    seed: int
    show_progress: bool

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
                "buyables_filter": "all route leaves present in buyables-stock by full inchikey",
            },
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


def build_training_records(
    all_routes: Sequence[dict[str, Any]],
    heldout_routes: dict[str, Sequence[dict[str, Any]]],
    buyables_stock: set[InchiKeyStr],
    config: TrainingSetBuildConfig,
) -> TrainingSetBuildResult:
    adapted_all_routes, all_adaptation = adapt_training_routes(
        all_routes,
        dataset="all",
        id_width=6,
        collect_reactions=config.holdout_mode == "reaction",
        show_progress=config.show_progress,
    )
    adapted_heldout_routes: dict[str, list[AdaptedTrainingRoute]] = {}
    heldout_adaptation: dict[str, AdaptationStatistics] = {}
    for dataset, routes in heldout_routes.items():
        adapted_heldout_routes[dataset], heldout_adaptation[dataset] = adapt_training_routes(
            routes,
            dataset=dataset,
            id_width=5,
            collect_reactions=config.holdout_mode == "reaction",
            show_progress=config.show_progress,
        )

    return build_training_records_from_adapted(
        all_routes=adapted_all_routes,
        all_adaptation=all_adaptation,
        heldout_routes=adapted_heldout_routes,
        heldout_adaptation=heldout_adaptation,
        buyables_stock=buyables_stock,
        config=config,
    )


def build_training_records_from_adapted(
    all_routes: Sequence[AdaptedTrainingRoute],
    all_adaptation: AdaptationStatistics,
    heldout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
    heldout_adaptation: Mapping[str, AdaptationStatistics] | None,
    buyables_stock: set[InchiKeyStr],
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
        buyables_stock=buyables_stock,
        val_fraction=config.val_fraction,
        seed=config.seed,
        stock_match_level=config.stock_match_level,
    )

    records: list[TrainingRouteRecord] = []
    for ordinal, route_signature in enumerate(sorted(routes_by_signature), start=1):
        route = routes_by_signature[route_signature]
        leaves = sorted(route.leaves, key=lambda leaf: leaf.inchikey)
        missing = sorted(leaf.inchikey for leaf in leaves if leaf.inchikey not in buyables_stock)
        n_buyable_leaves = len(leaves) - len(missing)
        records.append(
            TrainingRouteRecord(
                id=f"{config.route_prefix}-{config.release_name}-{ordinal:06d}",
                split=split_by_signature[route_signature],
                route_signature=route_signature,
                content_hash=route.get_content_hash(),
                buyables_solved=is_route_solved(route, buyables_stock, match_level=config.stock_match_level),
                n_leaves=len(leaves),
                n_buyable_leaves=n_buyable_leaves,
                missing_buyable_leaf_inchikeys=missing,
                route=route,
                sources=sources_by_signature[route_signature],
            )
        )

    summary = TrainingReleaseSummary(
        input_all_routes=all_adaptation.raw_routes,
        adaptation=AdaptationSummary(
            all=all_adaptation,
            heldout=heldout_adaptation or {},
        ),
        holdout=HoldoutSummary(
            route_signatures=len(heldout_route_signatures),
            reaction_signatures=len(heldout_reaction_signatures),
            exact_route_matches_excluded=skipped_route_holdout,
            reaction_excision=ReactionExcisionStatistics(
                source_routes_with_overlap=reaction_excision_source_routes,
                surviving_fragments=reaction_excision_fragments,
                fully_removed_source_routes=fully_removed_by_reaction_excision,
            ),
        ),
        duplicate_routes_removed=duplicate_routes,
        output=summarize_records(records),
    )

    return TrainingSetBuildResult(
        release_name=config.release_name,
        records=records,
        summary=summary,
    )


def assign_train_val_splits(
    routes_by_signature: dict[str, Route],
    buyables_stock: set[InchiKeyStr],
    val_fraction: float,
    seed: int,
    stock_match_level: InchiKeyLevel = InchiKeyLevel.FULL,
) -> dict[str, SplitName]:
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    groups: dict[tuple[int, bool, bool], list[str]] = defaultdict(list)
    for signature, route in routes_by_signature.items():
        key = (
            route.length,
            route.has_convergent_reaction,
            is_route_solved(route, buyables_stock, match_level=stock_match_level),
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


def summarize_records(records: Sequence[TrainingRouteRecord]) -> OutputSummary:
    training_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "val"]
    stock_constrained_training_records = [record for record in training_records if record.buyables_solved]
    stock_constrained_validation_records = [record for record in validation_records if record.buyables_solved]

    return OutputSummary(
        all_records=SplitCounts(
            total=len(records),
            training_split=len(training_records),
            validation_split=len(validation_records),
        ),
        stock_constrained_records=SplitCounts(
            total=len(stock_constrained_training_records) + len(stock_constrained_validation_records),
            training_split=len(stock_constrained_training_records),
            validation_split=len(stock_constrained_validation_records),
        ),
    )


def write_training_release(
    result: TrainingSetBuildResult,
    output_dir: Path,
    source_paths: Sequence[Path],
    parameters: TrainingReleaseParameters,
    source_root: Path | None = None,
) -> dict[str, Any]:
    release_dir = output_dir / result.release_name
    buyables_dir = release_dir / "buyables"
    release_dir.mkdir(parents=True, exist_ok=True)
    buyables_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "all": release_dir / "all.jsonl.gz",
        "train": release_dir / "train.jsonl.gz",
        "val": release_dir / "val.jsonl.gz",
        "buyables_all": buyables_dir / "all.jsonl.gz",
        "buyables_train": buyables_dir / "train.jsonl.gz",
        "buyables_val": buyables_dir / "val.jsonl.gz",
    }

    write_jsonl_gz(result.records, files["all"])
    write_jsonl_gz((record for record in result.records if record.split == "train"), files["train"])
    write_jsonl_gz((record for record in result.records if record.split == "val"), files["val"])
    write_jsonl_gz((record for record in result.records if record.buyables_solved), files["buyables_all"])
    write_jsonl_gz(
        (record for record in result.records if record.buyables_solved and record.split == "train"),
        files["buyables_train"],
    )
    write_jsonl_gz(
        (record for record in result.records if record.buyables_solved and record.split == "val"),
        files["buyables_val"],
    )

    manifest = build_training_manifest(
        release_name=result.release_name,
        files=files,
        source_paths=source_paths,
        source_root=source_root,
        output_root=release_dir,
        parameters=parameters,
        summary=result.summary,
    )
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_jsonl_gz(records: Iterable[TrainingRouteRecord], path: Path) -> int:
    return save_jsonl_gz((record.to_json_dict() for record in records), path)


def build_training_manifest(
    release_name: str,
    files: dict[str, Path],
    source_paths: Sequence[Path],
    source_root: Path | None,
    output_root: Path,
    parameters: TrainingReleaseParameters,
    summary: TrainingReleaseSummary,
) -> dict[str, Any]:
    def _format_path(path: Path, root: Path | None) -> str:
        if root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path)

    return {
        "retrocast_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "action": "scripts/paroutes/training-set-prep/create-training-release",
        "release_name": release_name,
        "parameters": parameters.to_manifest_dict(),
        "summary": summary.to_manifest_dict(),
        "source_files": [
            {"path": _format_path(path, source_root), "sha256": calculate_file_hash(path)}
            for path in source_paths
            if path.exists()
        ],
        "output_files": {
            name: {"path": _format_path(path, output_root), "sha256": calculate_file_hash(path)}
            for name, path in sorted(files.items())
        },
    }
