"""
Create a benchmark from retro* pickle containing routes as acceptable routes.

Usage:
    uv run scripts/curation/uspto-190/create-benchmark-from-pickle.py
    uv run scripts/curation/uspto-190/create-benchmark-from-pickle.py --check-buyables

Steps:
1. Load routes from retro* pickle (list of reaction SMILES lists)
2. Extract target SMILES from first step of each route
3. Convert to RetroStar format and cast to Route objects
4. Create Benchmark with stock='buyables-stock'

With --check-buyables: keeps only routes that terminate in buyables stock
Without --check-buyables: skips stock-termination filtering
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from retrocast.adapters.retrostar import RetroStarAdapter, RetroStarRoutePayload
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import InvalidSmilesError, RetroCastException
from retrocast.io import create_manifest, load_stock_file, save_benchmark
from retrocast.metrics import StockTerminationChecker, check_task_constraints
from retrocast.models import Benchmark, Route, StockTerminationConstraint, Target
from retrocast.typing import InChIKeyStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"
STOCK_DIR = BASE_DIR / "data" / "1-benchmarks" / "stocks"


def extract_target_smiles(first_reaction: str) -> str:
    """Extract target SMILES from first reaction (format: target>>precursors)."""
    if ">>" not in first_reaction:
        raise ValueError(f"Invalid reaction format (no >>): {first_reaction}")
    target = first_reaction.split(">>")[0].strip()
    if not target:
        raise ValueError(f"Empty target in reaction: {first_reaction}")
    return canonicalize_smiles(target)


def to_retrostar_format(reactions: list[str]) -> str:
    """Convert reactions to RetroStar format: prod1>0>react1|prod2>0>react2|..."""
    if not reactions:
        raise ValueError("Empty reaction list")
    formatted = []
    for rxn in reactions:
        if ">>" not in rxn:
            raise ValueError(f"Invalid reaction: {rxn}")
        prod, react = rxn.split(">>", 1)
        if not prod.strip() or not react.strip():
            raise ValueError(f"Empty product/reactants: {rxn}")
        formatted.append(f"{prod}>0>{react}")
    return "|".join(formatted)


def create_benchmark_from_pickle(
    pickle_path: Path,
    check_buyables: bool = False,
) -> None:
    """Create benchmark from retro* pickle with optional buyables validation."""
    # Load routes from pickle
    logger.info(f"Loading routes from {pickle_path}")
    with open(pickle_path, "rb") as f:
        routes: list[list[str]] = pickle.load(f)
    logger.info(f"Loaded {len(routes)} routes")

    # Load stock if validation is requested
    stock = None
    if check_buyables:
        stock_path = STOCK_DIR / "buyables-stock.csv.gz"
        logger.info(f"Loading buyables stock for validation from {stock_path}")
        stock = load_stock_file(stock_path)
        logger.info(f"Loaded {len(stock)} buyables molecules")

    # Process routes into benchmark targets
    adapter = RetroStarAdapter()
    benchmark_targets: dict[str, Target] = {}
    failed = 0
    stock_termination_failures = 0
    for idx, route_steps in enumerate(routes):
        target_id = f"USPTO-{idx + 1:03d}/{len(routes)}"

        try:
            # Extract target and convert route format
            if not route_steps:
                raise ValueError("Empty route")
            target_smiles = extract_target_smiles(route_steps[0])
            route_str = to_retrostar_format(route_steps)

            # Cast to Route object
            target_input = Target(id=target_id, smiles=target_smiles, inchikey=get_inchi_key(target_smiles))
            route = adapter.cast(
                RetroStarRoutePayload(route_str=route_str, route_cost=None),
                target=target_input,
            )

            # If checking buyables, filter out routes that do not terminate in stock.
            if check_buyables:
                if stock is None:
                    raise ValueError("stock must be loaded when check_buyables=True")
                if not satisfies_stock_termination(route, stock=stock, stock_name="buyables-stock"):
                    stock_termination_failures += 1
                    continue

            benchmark_target = Target(
                id=target_id,
                smiles=target_smiles,
                inchikey=target_input.inchikey,
                acceptable_routes=[route],
                annotations={"source": "retro-pickle"},
            )
            benchmark_targets[target_id] = benchmark_target

        except (ValueError, RetroCastException, InvalidSmilesError) as e:
            logger.warning(f"{target_id}: {e}")
            failed += 1

    n_success = len(benchmark_targets)
    logger.info(
        f"Converted {n_success}/{len(routes)} routes "
        f"({failed} failed, {stock_termination_failures} failed stock termination)"
    )

    # Log route length distribution
    length_counts: dict[int, int] = {}
    for target in benchmark_targets.values():
        if target.acceptable_routes:
            length = target.acceptable_routes[0].depth()
            length_counts[length] = length_counts.get(length, 0) + 1

    if length_counts:
        logger.info(f"Route length distribution: {dict(sorted(length_counts.items()))}")

    if check_buyables:
        if stock is None:
            raise ValueError("stock must be loaded when check_buyables=True")
        validate_acceptable_routes_satisfy_stock(
            targets=benchmark_targets,
            stock=stock,
            stock_name="buyables-stock",
        )

    # Create benchmark
    DEF_DIR.mkdir(parents=True, exist_ok=True)
    name = f"uspto-{n_success}"

    description = (
        f"USPTO-{n_success} benchmark with buyables-terminating routes from retro* pickle."
        if check_buyables
        else f"USPTO-{n_success} benchmark with routes from retro* pickle as acceptable routes."
    )
    benchmark = Benchmark(
        name=name,
        description=description,
        targets=benchmark_targets,
        default_constraints=[StockTerminationConstraint(stock="buyables-stock")],
        annotations={"source": "retro-pickle"},
    )

    # Save benchmark
    out_path = DEF_DIR / f"{name}.json.gz"
    logger.info(f"Writing benchmark with {n_success} targets to {out_path}")
    save_benchmark(benchmark, out_path)

    # Create and save manifest
    manifest = create_manifest(
        action="scripts/curation/uspto-190/create-benchmark-from-pickle",
        sources=[pickle_path],
        outputs=[(out_path, benchmark, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={"check_buyables": check_buyables},
        statistics={
            "n_targets": n_success,
            "n_failed": failed,
            "n_stock_termination_failures": stock_termination_failures,
        },
    )

    manifest_path = DEF_DIR / f"{name}.manifest.json"
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2, exclude_none=True))

    logger.info(f"Benchmark saved: {out_path}")
    logger.info(f"Manifest saved: {manifest_path}")


def validate_acceptable_routes_satisfy_stock(
    *,
    targets: dict[str, Target],
    stock: set[InChIKeyStr],
    stock_name: str,
) -> None:
    for target in targets.values():
        for route_index, route in enumerate(target.acceptable_routes, start=1):
            if not satisfies_stock_termination(route, stock=stock, stock_name=stock_name):
                raise ValueError(
                    f"{target.id}: acceptable route {route_index} fails declared stock constraint '{stock_name}'"
                )


def satisfies_stock_termination(route: Route, *, stock: set[InChIKeyStr], stock_name: str) -> bool:
    result = check_task_constraints(
        route,
        [StockTerminationConstraint(stock=stock_name)],
        [StockTerminationChecker(stocks={stock_name: stock})],
    )
    return not result.checks


def main():
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Create USPTO benchmark from retro* pickle")
    parser.add_argument(
        "--check-buyables",
        action="store_true",
        help="Keep only routes that terminate in buyables stock",
    )
    args = parser.parse_args()

    create_benchmark_from_pickle(
        pickle_path=BASE_DIR / "data" / "0-assets" / "routes_possible_test_hard.pkl",
        check_buyables=args.check_buyables,
    )


if __name__ == "__main__":
    main()
