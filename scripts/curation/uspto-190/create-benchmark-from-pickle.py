"""
Create a benchmark from retro* pickle containing routes as acceptable routes.

Usage:
    uv run scripts/curation/uspto-190/create-benchmark-from-pickle.py
    uv run scripts/curation/uspto-190/create-benchmark-from-pickle.py --check-buyables

Steps:
1. Load routes from retro* pickle (list of reaction SMILES lists)
2. Extract target SMILES from first step of each route
3. Convert to RetroStar format and cast to Route objects
4. Create BenchmarkSet with stock_name='buyables-stock'

With --check-buyables: validates routes are solvable by buyables stock
Without --check-buyables: skips validation (many USPTO routes aren't solvable by buyables)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import InvalidSmilesError, RetroCastException
from retrocast.io import create_manifest, load_stock_file, save_json_gz
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget, create_benchmark, create_benchmark_target
from retrocast.models.chem import TargetInput
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
    benchmark_targets: dict[str, BenchmarkTarget] = {}
    failed = 0
    unsolvable = 0

    for idx, route_steps in enumerate(routes):
        target_id = f"USPTO-{idx + 1:03d}/{len(routes)}"

        try:
            # Extract target and convert route format
            if not route_steps:
                raise ValueError("Empty route")
            target_smiles = extract_target_smiles(route_steps[0])
            route_str = to_retrostar_format(route_steps)

            # Cast to Route object
            target_input = TargetInput(id=target_id, smiles=target_smiles)
            cast_routes = list(adapter.cast({"succ": True, "routes": route_str}, target_input))

            if not cast_routes:
                raise ValueError("Adapter produced no routes")

            route = cast_routes[0]

            # If checking buyables, filter out unsolvable routes
            if check_buyables and not is_route_solved(route, stock):
                unsolvable += 1
                continue

            # Create benchmark target
            benchmark_target = create_benchmark_target(
                id=target_id,
                smiles=target_smiles,
                acceptable_routes=[route],
                metadata={"source": "retro-pickle"},
            )
            benchmark_targets[target_id] = benchmark_target

        except (ValueError, RetroCastException, InvalidSmilesError) as e:
            logger.warning(f"{target_id}: {e}")
            failed += 1

    n_success = len(benchmark_targets)
    logger.info(f"Converted {n_success}/{len(routes)} routes ({failed} failed, {unsolvable} unsolvable)")

    # Log route length distribution
    length_counts: dict[int, int] = {}
    for target in benchmark_targets.values():
        if target.acceptable_routes:  # type: ignore
            length = target.acceptable_routes[0].length  # type: ignore
            length_counts[length] = length_counts.get(length, 0) + 1

    if length_counts:
        logger.info(f"Route length distribution: {dict(sorted(length_counts.items()))}")

    # Create benchmark
    DEF_DIR.mkdir(parents=True, exist_ok=True)
    name = f"uspto-{n_success}"

    if check_buyables:
        # All routes have been pre-filtered for solvability, so validation will pass
        description = f"USPTO-{n_success} benchmark with buyables-solvable routes from retro* pickle."
        benchmark = create_benchmark(
            name=name,
            description=description,
            stock=stock,
            stock_name="buyables-stock",
            targets=benchmark_targets,
        )
    else:
        # Set stock_name but skip validation (many USPTO routes not solvable by buyables)
        description = f"USPTO-{n_success} benchmark with routes from retro* pickle as acceptable routes."
        benchmark = BenchmarkSet(
            name=name,
            description=description,
            stock_name="buyables-stock",
            targets=benchmark_targets,
        )

    # Save benchmark
    out_path = DEF_DIR / f"{name}.json.gz"
    logger.info(f"Writing benchmark with {n_success} targets to {out_path}")
    save_json_gz(benchmark, out_path)

    # Create and save manifest
    manifest = create_manifest(
        action="scripts/curation/uspto-190/create-benchmark-from-pickle",
        sources=[pickle_path],
        outputs=[(out_path, benchmark, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={"check_buyables": check_buyables},
        statistics={"n_targets": n_success, "n_failed": failed, "n_unsolvable": unsolvable},
    )

    manifest_path = DEF_DIR / f"{name}.manifest.json"
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Benchmark saved: {out_path}")
    logger.info(f"Manifest saved: {manifest_path}")


def main():
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Create USPTO benchmark from retro* pickle")
    parser.add_argument(
        "--check-buyables",
        action="store_true",
        help="Validate that acceptable routes are solvable by buyables stock",
    )
    args = parser.parse_args()

    create_benchmark_from_pickle(
        pickle_path=BASE_DIR / "data" / "0-assets" / "routes_possible_test_hard.pkl",
        check_buyables=args.check_buyables,
    )


if __name__ == "__main__":
    main()
