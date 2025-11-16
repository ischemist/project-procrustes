"""
Creates train/val splits from PaRoutes data.

This script loads all processed routes, removes routes that appear in the n1
and n5 test sets (matched by route signature), and splits the remaining routes
into 90% train and 10% validation partitions.

Two versions are created:
1. Standard split: excludes routes with identical signatures to test set
2. Strict split: excludes routes with any overlapping reactions with test set

Usage:
    uv run scripts/paroutes/2-create-train-val.py
"""

from pathlib import Path

from retrocast.curation import (
    create_manifest,
    filter_routes_by_reaction_overlap,
    filter_routes_by_signature,
    get_reaction_signatures,
    get_route_signatures,
    split_routes,
)
from retrocast.io import load_routes, save_json, save_routes
from retrocast.schemas import Route
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "paroutes" / "processed"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def save_partition(
    routes: dict[str, list[Route]],
    name: str,
    source_file: Path,
    output_dir: Path,
    split_ratio: float,
    seed: int,
) -> None:
    """Save a partition with its manifest."""
    output_file = output_dir / f"{name}.json.gz"
    manifest_file = output_dir / f"{name}-manifest.json"

    logger.info(f"Saving {name} to {output_file.relative_to(BASE_DIR)}...")
    save_routes(routes, output_file)

    stats = {
        "n_targets": len(routes),
        "n_routes": sum(len(r) for r in routes.values()),
        "split_ratio": split_ratio,
        "random_seed": seed,
    }
    manifest = create_manifest(name, source_file, output_file, routes, stats)
    save_json(manifest, manifest_file)


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
    n1_route_sigs = get_route_signatures(n1_routes)
    n5_route_sigs = get_route_signatures(n5_routes)
    test_route_sigs = n1_route_sigs | n5_route_sigs
    logger.info(f"Found {len(test_route_sigs)} unique test route signatures")

    n1_reaction_sigs = get_reaction_signatures(n1_routes)
    n5_reaction_sigs = get_reaction_signatures(n5_routes)
    test_reaction_sigs = n1_reaction_sigs | n5_reaction_sigs
    logger.info(f"Found {len(test_reaction_sigs)} unique test reaction signatures")

    n_routes_total = sum(len(r) for r in all_routes.values())

    # === Standard split: exclude identical routes ===
    logger.info("\n=== Creating standard split (exclude identical routes) ===")
    filtered_standard = filter_routes_by_signature(all_routes, test_route_sigs)
    n_filtered_standard = sum(len(r) for r in filtered_standard.values())
    logger.info(f"Removed {n_routes_total - n_filtered_standard} routes, {n_filtered_standard} remaining")

    train_standard, val_standard = split_routes(filtered_standard, TRAIN_RATIO, RANDOM_SEED)
    logger.info(f"Train: {len(train_standard)} targets, {sum(len(r) for r in train_standard.values())} routes")
    logger.info(f"Val: {len(val_standard)} targets, {sum(len(r) for r in val_standard.values())} routes")

    save_partition(train_standard, "train-routes", all_routes_file, PROCESSED_DIR, TRAIN_RATIO, RANDOM_SEED)
    save_partition(val_standard, "val-routes", all_routes_file, PROCESSED_DIR, 1 - TRAIN_RATIO, RANDOM_SEED)

    # === Strict split: exclude routes with any reaction overlap ===
    logger.info("\n=== Creating strict split (exclude routes with reaction overlap) ===")
    filtered_strict = filter_routes_by_reaction_overlap(all_routes, test_reaction_sigs)
    n_filtered_strict = sum(len(r) for r in filtered_strict.values())
    logger.info(f"Removed {n_routes_total - n_filtered_strict} routes, {n_filtered_strict} remaining")

    train_strict, val_strict = split_routes(filtered_strict, TRAIN_RATIO, RANDOM_SEED)
    logger.info(f"Train: {len(train_strict)} targets, {sum(len(r) for r in train_strict.values())} routes")
    logger.info(f"Val: {len(val_strict)} targets, {sum(len(r) for r in val_strict.values())} routes")

    save_partition(train_strict, "train-routes-strict", all_routes_file, PROCESSED_DIR, TRAIN_RATIO, RANDOM_SEED)
    save_partition(val_strict, "val-routes-strict", all_routes_file, PROCESSED_DIR, 1 - TRAIN_RATIO, RANDOM_SEED)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
