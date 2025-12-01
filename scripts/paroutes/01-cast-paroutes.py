"""
Casts raw PaRoutes n1/n5 JSON files into the canonical BenchmarkSet format.

Usage:
    uv run scripts/paroutes/01-cast-paroutes.py
    uv run scripts/paroutes/01-cast-paroutes.py --check-buyables
    uv run scripts/paroutes/01-cast-paroutes.py --prune-intermediates
    uv run scripts/paroutes/01-cast-paroutes.py --check-buyables --prune-intermediates
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast import adapt_single_route
from retrocast.chem import canonicalize_smiles
from retrocast.curation.generators import generate_pruned_routes
from retrocast.io import create_manifest, load_raw_paroutes_list, load_stock_file, save_json_gz
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import create_benchmark, create_benchmark_target
from retrocast.models.chem import TargetInput
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "0-assets" / "paroutes"
DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"
stock_dir = BASE_DIR / "data" / "1-benchmarks" / "stocks"

DATASETS = ["n1", "n5"]


def process_dataset(name: str, check_buyables: bool = False, prune_intermediates: bool = False):
    raw_path = RAW_DIR / f"{name}-routes.json.gz"

    # Determine output filename based on options
    suffix_parts = []
    if check_buyables:
        suffix_parts.append("buyables")
    if prune_intermediates:
        suffix_parts.append("pruned")
    suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
    out_path = DEF_DIR / f"paroutes-{name}-full{suffix}.json.gz"

    logger.info(f"Processing {name}... (check_buyables={check_buyables}, prune_intermediates={prune_intermediates})")
    raw_list = load_raw_paroutes_list(raw_path)

    # Load the appropriate stock for this dataset
    if check_buyables:
        stock_path = stock_dir / "buyables-stock.csv.gz"
        stock_name_str = "buyables-stock"
    else:
        stock_path = stock_dir / f"{name}-stock.csv.gz"
        stock_name_str = f"{name}-stock"
    stock_for_validation = load_stock_file(stock_path)

    targets = {}
    failures = 0
    unsolved = 0
    total_pruned_routes = 0
    targets_with_pruning = 0

    for i, raw_item in tqdm(enumerate(raw_list), total=len(raw_list), desc=f"Casting {name}"):
        # Generate stable ID based on index
        # n5-00001, n5-00002...
        target_id = f"{name}-{i + 1:05d}"

        # 1. Canonicalize SMILES
        smiles = canonicalize_smiles(raw_item["smiles"])

        # 2. Adapt the Route
        # (We construct a temporary TargetIdentity for the adapter)
        target_input = TargetInput(id=target_id, smiles=smiles)
        route = adapt_single_route(raw_item, target_input, "paroutes")

        if not route:
            failures += 1
            continue

        # Check if route is solved (only if check_buyables is enabled)
        if check_buyables:
            is_solved = is_route_solved(route, stock_for_validation)
            if not is_solved:
                unsolved += 1
                continue

        # 3. Optionally generate pruned route variants
        acceptable_routes = [route]  # Always include the original route as primary
        if prune_intermediates:
            pruned_variants = generate_pruned_routes(route, stock_for_validation)
            # pruned_variants includes the original route, so we use it directly
            if len(pruned_variants) > 1:
                acceptable_routes = pruned_variants
                targets_with_pruning += 1
                total_pruned_routes += len(pruned_variants) - 1  # Count only the additional variants

        # 4. Create BenchmarkTarget using official constructor
        # (canonicalizes SMILES and computes InChIKey)
        target = create_benchmark_target(
            id=target_id,
            smiles=smiles,
            acceptable_routes=acceptable_routes,
        )

        targets[target_id] = target

    # Create benchmark with validation
    # This will validate that all acceptable routes are solvable with the stock
    benchmark = create_benchmark(
        name=f"paroutes-{name}-full{suffix}",
        description=f"Full raw import of PaRoutes {name} set.",
        stock=stock_for_validation,
        stock_name=stock_name_str,
        targets=targets,
    )

    logger.info(f"Created {len(benchmark.targets)} targets. {failures} failed.")
    if check_buyables:
        logger.info(f"{unsolved} routes were unsolved and excluded.")
    if prune_intermediates:
        logger.info(f"{targets_with_pruning} targets have pruned variants ({total_pruned_routes} additional routes).")
    save_json_gz(benchmark, out_path)

    manifest_path = DEF_DIR / f"paroutes-{name}-full{suffix}.manifest.json"
    statistics = {"n_targets": len(benchmark.targets), "n_failures": failures}
    if check_buyables:
        statistics["n_unsolved"] = unsolved
    if prune_intermediates:
        statistics["n_targets_with_pruning"] = targets_with_pruning
        statistics["n_total_pruned_routes"] = total_pruned_routes
        statistics["avg_pruned_routes_per_target"] = (
            total_pruned_routes / targets_with_pruning if targets_with_pruning > 0 else 0
        )

    manifest = create_manifest(
        action="scripts/paroutes/01-cast-paroutes",
        sources=[raw_path],
        outputs=[(out_path, benchmark, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={"dataset": name, "check_buyables": check_buyables, "prune_intermediates": prune_intermediates},
        statistics=statistics,
    )
    # Save Manifest (plain JSON, not gzipped, so it's readable)
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Manifest saved to {manifest_path}")


def main():
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Cast raw PaRoutes data to BenchmarkSet format")
    parser.add_argument(
        "--check-buyables",
        action="store_true",
        help="Filter routes to only include those solved by buyables stock (default: disabled)",
    )
    parser.add_argument(
        "--prune-intermediates",
        action="store_true",
        help="Generate pruned route variants by treating stock-available intermediates as leaves (default: disabled)",
    )
    args = parser.parse_args()

    DEF_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        process_dataset(ds, check_buyables=args.check_buyables, prune_intermediates=args.prune_intermediates)


if __name__ == "__main__":
    main()
