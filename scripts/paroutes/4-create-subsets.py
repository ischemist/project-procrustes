"""
Creates evaluation subsets from n5-routes (and n1-routes for convergent).

Generates subsets separated by route type:
- Linear routes: routes with only sequential reactions
- Convergent routes: routes with at least one convergent reaction

Each configuration is sampled with 3 different seeds for reproducibility testing.

Usage:
    uv run scripts/paroutes/4-create-subsets.py
"""

from pathlib import Path

from retrocast.curation import (
    create_manifest,
    filter_routes_by_convergence,
    merge_routes,
    sample_routes_by_length,
    sample_routes_random,
)
from retrocast.io import load_routes, save_json, save_routes
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "paroutes" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "paroutes" / "subsets"

# Seeds for reproducibility
SEEDS = [42, 10302025, 3172026]


def main() -> None:
    """Main script execution."""
    n5_file = PROCESSED_DIR / "n5-routes.json.gz"
    n1_file = PROCESSED_DIR / "n1-routes.json.gz"

    logger.info(f"Loading routes from {n5_file.relative_to(BASE_DIR)}...")
    n5_routes = load_routes(n5_file)
    logger.info(f"Loading routes from {n1_file.relative_to(BASE_DIR)}...")
    n1_routes = load_routes(n1_file)

    # Separate by convergence
    n5_linear = filter_routes_by_convergence(n5_routes, keep_convergent=False)
    n5_convergent = filter_routes_by_convergence(n5_routes, keep_convergent=True)
    n1_convergent = filter_routes_by_convergence(n1_routes, keep_convergent=True)

    # Merge n5 and n1 convergent routes for larger pool
    all_convergent = merge_routes(n5_convergent, n1_convergent)

    # Show distributions
    logger.info(f"N5 linear routes: {sum(len(r) for r in n5_linear.values())}")
    logger.info(f"N5 convergent routes: {sum(len(r) for r in n5_convergent.values())}")
    logger.info(f"N1 convergent routes: {sum(len(r) for r in n1_convergent.values())}")
    logger.info(f"Combined convergent routes: {sum(len(r) for r in all_convergent.values())}")

    # Show convergent depth distribution
    conv_by_depth: dict[int, int] = {}
    for route_list in all_convergent.values():
        for route in route_list:
            d = route.length
            conv_by_depth[d] = conv_by_depth.get(d, 0) + 1
    logger.info(f"Combined convergent by depth: {dict(sorted(conv_by_depth.items()))}")

    # Define subset configurations
    subset_configs = [
        {
            "name": "random-100",
            "source": n5_routes,
            "source_files": [n5_file],
            "type": "random",
            "n": 100,
        },
        {
            "name": "random-500",
            "source": n5_routes,
            "source_files": [n5_file],
            "type": "random",
            "n": 500,
        },
        {
            "name": "linear-300",
            "source": n5_linear,
            "source_files": [n5_file],
            "type": "stratified",
            "length_counts": {2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 50},
        },
        {
            "name": "linear-600",
            "source": n5_linear,
            "source_files": [n5_file],
            "type": "stratified",
            "length_counts": {2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100},
        },
        {
            "name": "convergent-150",
            "source": all_convergent,
            "source_files": [n5_file, n1_file],
            "type": "stratified",
            "length_counts": {2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25},
        },
        {
            "name": "convergent-250",
            "source": all_convergent,
            "source_files": [n5_file, n1_file],
            "type": "stratified",
            "length_counts": {2: 50, 3: 50, 4: 50, 5: 50, 6: 50},
        },
    ]

    all_manifests = []

    for seed in SEEDS:
        seed_dir = OUTPUT_DIR / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== Processing seed {seed} ===")

        for config in subset_configs:
            name = config["name"]
            logger.info(f"Creating {name}...")

            # Sample routes
            if config["type"] == "random":
                sampled = sample_routes_random(config["source"], config["n"], seed)
            else:  # stratified
                sampled = sample_routes_by_length(config["source"], config["length_counts"], seed)

            # Save routes
            output_file = seed_dir / f"{name}.json.gz"
            save_routes(sampled, output_file)

            # Create manifest
            n_sampled = sum(len(r) for r in sampled.values())
            stats = {
                "n_targets": len(sampled),
                "n_routes": n_sampled,
                "seed": seed,
                "sampling_type": config["type"],
            }
            if config["type"] == "random":
                stats["requested_n"] = config["n"]
            else:
                stats["length_counts"] = config["length_counts"]
                stats["route_type"] = "linear" if "linear" in name else "convergent"

            # For manifest, use primary source file (n5)
            primary_source = config["source_files"][0]
            manifest = create_manifest(name, primary_source, output_file, sampled, stats)
            manifest["seed"] = seed
            if len(config["source_files"]) > 1:
                manifest["additional_sources"] = [f.name for f in config["source_files"][1:]]
            all_manifests.append(manifest)

            logger.info(f"  Saved {n_sampled} routes to {output_file.relative_to(BASE_DIR)}")

    # Save combined manifest
    combined_manifest_file = OUTPUT_DIR / "manifest.json"
    save_json({"subsets": all_manifests}, combined_manifest_file)
    logger.info(f"\nSaved combined manifest to {combined_manifest_file.relative_to(BASE_DIR)}")
    logger.info(f"Total subsets created: {len(all_manifests)}")


if __name__ == "__main__":
    main()
