"""
Creates evaluation subsets from n5-routes.

Generates multiple subset configurations with different sampling strategies:
- Random sampling (100 and 500 routes)
- Stratified by route length

Each configuration is sampled with 3 different seeds for reproducibility testing.

Usage:
    uv run scripts/paroutes/4-create-subsets.py
"""

from pathlib import Path

from retrocast.curation import create_manifest, sample_routes_by_length, sample_routes_random
from retrocast.io import load_routes, save_json, save_routes
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "paroutes" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "paroutes" / "subsets"

# Seeds for reproducibility
SEEDS = [42, 10302025, 3172026]

# Subset configurations
SUBSET_CONFIGS = [
    {
        "name": "subset-1-random-100",
        "type": "random",
        "n": 100,
    },
    {
        "name": "subset-2-random-500",
        "type": "random",
        "n": 500,
    },
    {
        "name": "subset-3-stratified-100",
        "type": "stratified",
        "length_counts": {2: 25, 4: 25, 6: 25, 8: 25},
    },
    {
        "name": "subset-4-stratified-350",
        "type": "stratified",
        "length_counts": {2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 50},
    },
    {
        "name": "subset-5-stratified-700",
        "type": "stratified",
        "length_counts": {2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100},
    },
]


def main() -> None:
    """Main script execution."""
    source_file = PROCESSED_DIR / "n5-routes.json.gz"

    logger.info(f"Loading routes from {source_file.relative_to(BASE_DIR)}...")
    all_routes = load_routes(source_file)
    n_total = sum(len(r) for r in all_routes.values())
    logger.info(f"Loaded {n_total} routes from {len(all_routes)} targets")

    # Show route depth distribution
    depth_counts: dict[int, int] = {}
    for route_list in all_routes.values():
        for route in route_list:
            depth = route.depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
    logger.info(f"Route depth distribution: {dict(sorted(depth_counts.items()))}")

    all_manifests = []

    for seed in SEEDS:
        seed_dir = OUTPUT_DIR / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== Processing seed {seed} ===")

        for config in SUBSET_CONFIGS:
            name = config["name"]
            logger.info(f"Creating {name}...")

            # Sample routes
            if config["type"] == "random":
                sampled = sample_routes_random(all_routes, config["n"], seed)
            else:  # stratified
                sampled = sample_routes_by_length(all_routes, config["length_counts"], seed)

            # Save routes
            output_file = seed_dir / f"{name}.json.gz"
            save_routes(sampled, output_file)

            # Create manifest
            n_sampled = sum(len(r) for r in sampled.values())
            stats = {
                "n_targets": len(sampled),
                "n_routes": n_sampled,
                "sampling_type": config["type"],
            }
            if config["type"] == "random":
                stats["requested_n"] = config["n"]
            else:
                stats["length_counts"] = config["length_counts"]

            manifest = create_manifest(name, source_file, output_file, sampled, stats)
            manifest["seed"] = seed  # Add seed to top level for easier querying
            all_manifests.append(manifest)

            logger.info(f"  Saved {n_sampled} routes to {output_file.relative_to(BASE_DIR)}")

    # Save combined manifest
    combined_manifest_file = OUTPUT_DIR / "manifest.json"
    save_json({"subsets": all_manifests}, combined_manifest_file)
    logger.info(f"\nSaved combined manifest to {combined_manifest_file.relative_to(BASE_DIR)}")
    logger.info(f"Total subsets created: {len(all_manifests)}")


if __name__ == "__main__":
    main()
