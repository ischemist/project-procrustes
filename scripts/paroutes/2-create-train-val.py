"""
Creates train/val splits from PaRoutes data.

This script loads all processed routes, removes routes that appear in the n1
and n5 test sets (matched by route signature), and splits the remaining routes
into 90% train and 10% validation partitions.

Usage:
    uv run scripts/paroutes/2-create-train-val.py
"""

import random
from pathlib import Path

from retrocast.curation import create_manifest
from retrocast.io import load_routes, save_json, save_routes
from retrocast.schemas import Route
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "paroutes" / "processed"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def get_route_signatures(routes: dict[str, list[Route]]) -> set[str]:
    """Extract all route signatures from routes dict."""
    signatures = set()
    for route_list in routes.values():
        for route in route_list:
            signatures.add(route.get_signature())
    return signatures


def filter_routes_by_signature(routes: dict[str, list[Route]], exclude_signatures: set[str]) -> dict[str, list[Route]]:
    """Remove routes whose signatures are in exclude set."""
    filtered = {}
    for target_id, route_list in routes.items():
        kept_routes = [r for r in route_list if r.get_signature() not in exclude_signatures]
        if kept_routes:
            filtered[target_id] = kept_routes
    return filtered


def split_routes(
    routes: dict[str, list[Route]], train_ratio: float, seed: int
) -> tuple[dict[str, list[Route]], dict[str, list[Route]]]:
    """Split routes dict into train/val partitions."""
    target_ids = list(routes.keys())
    random.seed(seed)
    random.shuffle(target_ids)

    split_idx = int(len(target_ids) * train_ratio)
    train_ids = target_ids[:split_idx]
    val_ids = target_ids[split_idx:]

    train_routes = {tid: routes[tid] for tid in train_ids}
    val_routes = {tid: routes[tid] for tid in val_ids}

    return train_routes, val_routes


def main() -> None:
    """Main script execution."""
    # Load all processed routes
    all_routes_file = PROCESSED_DIR / "all-routes.json.gz"
    n1_routes_file = PROCESSED_DIR / "n1-routes.json.gz"
    n5_routes_file = PROCESSED_DIR / "n5-routes.json.gz"

    logger.info("Loading processed routes...")
    all_routes = load_routes(all_routes_file)
    n1_routes = load_routes(n1_routes_file)
    n5_routes = load_routes(n5_routes_file)

    # Extract signatures from test sets
    logger.info("Extracting test set signatures...")
    n1_signatures = get_route_signatures(n1_routes)
    n5_signatures = get_route_signatures(n5_routes)
    test_signatures = n1_signatures | n5_signatures
    logger.info(
        f"Found {len(test_signatures)} unique test set signatures (n1: {len(n1_signatures)}, n5: {len(n5_signatures)})"
    )

    # Filter out test routes
    logger.info("Filtering out test set routes...")
    n_routes_before = sum(len(r) for r in all_routes.values())
    filtered_routes = filter_routes_by_signature(all_routes, test_signatures)
    n_routes_after = sum(len(r) for r in filtered_routes.values())
    logger.info(f"Removed {n_routes_before - n_routes_after} routes, {n_routes_after} remaining")

    # Split into train/val
    logger.info(f"Splitting routes {int(TRAIN_RATIO * 100)}:{int((1 - TRAIN_RATIO) * 100)}...")
    train_routes, val_routes = split_routes(filtered_routes, TRAIN_RATIO, RANDOM_SEED)
    logger.info(f"Train: {len(train_routes)} targets, {sum(len(r) for r in train_routes.values())} routes")
    logger.info(f"Val: {len(val_routes)} targets, {sum(len(r) for r in val_routes.values())} routes")

    # Save train partition
    train_file = PROCESSED_DIR / "train-routes.json.gz"
    train_manifest_file = PROCESSED_DIR / "train-routes-manifest.json"
    logger.info(f"Saving train routes to {train_file.relative_to(BASE_DIR)}...")
    save_routes(train_routes, train_file)

    train_stats = {
        "n_targets": len(train_routes),
        "n_routes": sum(len(r) for r in train_routes.values()),
        "split_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
    }
    train_manifest = create_manifest("train-routes", all_routes_file, train_file, train_routes, train_stats)
    save_json(train_manifest, train_manifest_file)

    # Save val partition
    val_file = PROCESSED_DIR / "val-routes.json.gz"
    val_manifest_file = PROCESSED_DIR / "val-routes-manifest.json"
    logger.info(f"Saving val routes to {val_file.relative_to(BASE_DIR)}...")
    save_routes(val_routes, val_file)

    val_stats = {
        "n_targets": len(val_routes),
        "n_routes": sum(len(r) for r in val_routes.values()),
        "split_ratio": 1 - TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
    }
    val_manifest = create_manifest("val-routes", all_routes_file, val_file, val_routes, val_stats)
    save_json(val_manifest, val_manifest_file)

    logger.info("Done!")


if __name__ == "__main__":
    main()
