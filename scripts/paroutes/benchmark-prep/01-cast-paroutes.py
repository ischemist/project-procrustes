"""
Casts raw PaRoutes n1/n5 JSON files into the canonical BenchmarkSet format.

Usage:
    uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py
    uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --check-buyables
    uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --prune-intermediates
    uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --check-buyables --prune-intermediates
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast import adapt_route, get_adapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.curation.generators import generate_pruned_routes
from retrocast.io import create_manifest, load_raw_paroutes_list, load_stock_file, save_benchmark
from retrocast.metrics import TaskConstraintChecker
from retrocast.models import Benchmark, Route, Target, TaskConstraints
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "retrocast"
RAW_DIR = BASE_DIR / "0-assets" / "paroutes"
DEF_DIR = BASE_DIR / "1-benchmarks" / "definitions"
stock_dir = BASE_DIR / "1-benchmarks" / "stocks"

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
    stock_termination_failures = 0
    total_pruned_routes = 0
    targets_with_pruning = 0
    stock_checker = TaskConstraintChecker.stock_termination(stock=stock_for_validation, stock_name=stock_name_str)

    for i, raw_item in tqdm(enumerate(raw_list), total=len(raw_list), desc=f"Casting {name}"):
        # Generate stable ID based on index
        # n5-00001, n5-00002...
        target_id = f"{name}-{i + 1:05d}"

        # 1. Canonicalize SMILES
        smiles = canonicalize_smiles(raw_item["smiles"])

        # 2. Adapt the Route
        # (We construct a temporary TargetIdentity for the adapter)
        target_input = Target(id=target_id, smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
        route = adapt_route(raw_item, get_adapter("paroutes"), target=target_input)

        if not route:
            failures += 1
            continue

        # Optionally filter to routes that terminate in buyables stock.
        if check_buyables and not satisfies_stock_termination(
            route,
            stock_checker=stock_checker,
            stock_name=stock_name_str,
        ):
            stock_termination_failures += 1
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

        # 4. Create Target and preserve the curation invariant that every
        # acceptable route satisfies the benchmark's stock constraint.
        target = Target(
            id=target_id,
            smiles=SmilesStr(smiles),
            inchikey=InChIKeyStr(get_inchi_key(smiles)),
            acceptable_routes=acceptable_routes,
        )
        validate_acceptable_routes_satisfy_stock(
            target_id=target_id,
            routes=target.acceptable_routes,
            stock_checker=stock_checker,
            stock_name=stock_name_str,
        )

        targets[target_id] = target

    benchmark = Benchmark(
        name=f"paroutes-{name}-full{suffix}",
        description=f"Full raw import of PaRoutes {name} set.",
        targets=targets,
        default_constraints=TaskConstraints(stock=stock_name_str),
    )

    logger.info(f"Created {len(benchmark.targets)} targets. {failures} failed.")
    if check_buyables:
        logger.info(f"{stock_termination_failures} routes failed buyables stock termination and were excluded.")
    if prune_intermediates:
        logger.info(f"{targets_with_pruning} targets have pruned variants ({total_pruned_routes} additional routes).")
    save_benchmark(benchmark, out_path)

    manifest_path = DEF_DIR / f"paroutes-{name}-full{suffix}.manifest.json"
    statistics: dict[str, int | float] = {"n_targets": len(benchmark.targets), "n_failures": failures}
    if check_buyables:
        statistics["n_stock_termination_failures"] = stock_termination_failures
    if prune_intermediates:
        statistics["n_targets_with_pruning"] = targets_with_pruning
        statistics["n_total_pruned_routes"] = total_pruned_routes
        statistics["avg_pruned_routes_per_target"] = (
            total_pruned_routes / targets_with_pruning if targets_with_pruning > 0 else 0
        )

    manifest = create_manifest(
        action="scripts/paroutes/benchmark-prep/01-cast-paroutes",
        sources=[raw_path],
        outputs=[(out_path, benchmark, "benchmark")],
        root_dir=BASE_DIR,
        parameters={"dataset": name, "check_buyables": check_buyables, "prune_intermediates": prune_intermediates},
        statistics=statistics,
    )
    # Save Manifest (plain JSON, not gzipped, so it's readable)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Manifest saved to {manifest_path}")


def validate_acceptable_routes_satisfy_stock(
    *,
    target_id: str,
    routes: list[Route],
    stock_checker: TaskConstraintChecker,
    stock_name: str,
) -> None:
    for route_index, route in enumerate(routes, start=1):
        if not satisfies_stock_termination(route, stock_checker=stock_checker, stock_name=stock_name):
            raise ValueError(
                f"{target_id}: acceptable route {route_index} fails declared stock constraint '{stock_name}'"
            )


def satisfies_stock_termination(route: Route, *, stock_checker: TaskConstraintChecker, stock_name: str) -> bool:
    return not stock_checker.check_route(route, TaskConstraints(stock=stock_name)).checks


def main():
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Cast raw PaRoutes data to BenchmarkSet format")
    parser.add_argument(
        "--check-buyables",
        action="store_true",
        help="Filter routes to only include those terminating in buyables stock (default: disabled)",
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
