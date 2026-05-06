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

from retrocast import __version__, adapt_single_route
from retrocast.chem import InchiKeyLevel
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
    statistics: dict[str, Any]


@dataclass(frozen=True)
class AdaptedTrainingRoute:
    route: Route
    route_signature: str
    reaction_signatures: set[ReactionSignature]
    source: RawRouteSource


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
) -> tuple[list[AdaptedTrainingRoute], int]:
    adapted_routes: list[AdaptedTrainingRoute] = []
    skipped_adaptation = 0
    routes = progress_iter(raw_routes, desc=f"adapt {dataset}", enabled=show_progress)
    for idx, raw_route in enumerate(routes):
        target = TargetInput(id=f"{dataset}-{idx + 1:0{id_width}d}", smiles=raw_route["smiles"])
        route = adapt_single_route(raw_route, target, "paroutes")
        if route is None:
            skipped_adaptation += 1
            continue
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
    return adapted_routes, skipped_adaptation


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


def build_training_records(
    all_routes: Sequence[dict[str, Any]],
    heldout_routes: dict[str, Sequence[dict[str, Any]]],
    buyables_stock: set[InchiKeyStr],
    config: TrainingSetBuildConfig,
) -> TrainingSetBuildResult:
    adapted_all_routes, skipped_adaptation = adapt_training_routes(
        all_routes,
        dataset="all",
        id_width=6,
        collect_reactions=config.holdout_mode == "reaction",
        show_progress=config.show_progress,
    )
    adapted_heldout_routes: dict[str, list[AdaptedTrainingRoute]] = {}
    for dataset, routes in heldout_routes.items():
        adapted_heldout_routes[dataset], _ = adapt_training_routes(
            routes,
            dataset=dataset,
            id_width=5,
            collect_reactions=config.holdout_mode == "reaction",
            show_progress=config.show_progress,
        )

    return build_training_records_from_adapted(
        all_routes=adapted_all_routes,
        raw_all_routes_count=len(all_routes),
        skipped_adaptation=skipped_adaptation,
        heldout_routes=adapted_heldout_routes,
        buyables_stock=buyables_stock,
        config=config,
    )


def build_training_records_from_adapted(
    all_routes: Sequence[AdaptedTrainingRoute],
    raw_all_routes_count: int,
    skipped_adaptation: int,
    heldout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
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
    skipped_reaction_holdout = 0
    duplicate_routes = 0

    for adapted_route in all_routes:
        route = adapted_route.route
        route_signature = adapted_route.route_signature
        if route_signature in heldout_route_signatures:
            skipped_route_holdout += 1
            continue

        if config.holdout_mode == "reaction" and adapted_route.reaction_signatures & heldout_reaction_signatures:
            skipped_reaction_holdout += 1
            continue

        if route_signature in routes_by_signature:
            duplicate_routes += 1
        else:
            routes_by_signature[route_signature] = route

        sources_by_signature[route_signature].append(adapted_route.source)

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

    statistics = summarize_records(records)
    statistics.update(
        {
            "raw_all_routes": raw_all_routes_count,
            "heldout_route_signatures": len(heldout_route_signatures),
            "heldout_reaction_signatures": len(heldout_reaction_signatures),
            "skipped_adaptation": skipped_adaptation,
            "skipped_route_holdout": skipped_route_holdout,
            "skipped_reaction_holdout": skipped_reaction_holdout,
            "duplicate_routes_removed": duplicate_routes,
            "val_fraction": config.val_fraction,
            "seed": config.seed,
        }
    )

    return TrainingSetBuildResult(
        release_name=config.release_name,
        records=records,
        statistics=statistics,
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


def summarize_records(records: Sequence[TrainingRouteRecord]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "n_records": len(records),
        "splits": {},
        "buyables": {},
    }
    for split in ("train", "val"):
        split_records = [record for record in records if record.split == split]
        buyables_records = [record for record in split_records if record.buyables_solved]
        counts["splits"][split] = len(split_records)
        counts["buyables"][split] = len(buyables_records)
    counts["buyables"]["all"] = sum(record.buyables_solved for record in records)
    return counts


def write_training_release(
    result: TrainingSetBuildResult,
    output_dir: Path,
    source_paths: Sequence[Path],
    parameters: dict[str, Any],
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
        statistics=result.statistics,
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
    parameters: dict[str, Any],
    statistics: dict[str, Any],
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
        "parameters": parameters,
        "statistics": statistics,
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
