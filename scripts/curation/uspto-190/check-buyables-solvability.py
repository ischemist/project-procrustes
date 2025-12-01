"""
Check how many acceptable routes from the USPTO-190 pickle benchmark are solvable with buyables stock.

Usage:
    uv run scripts/curation/uspto-190/check-buyables-solvability.py
"""

from __future__ import annotations

from pathlib import Path

from retrocast.io import load_benchmark, load_stock_file
from retrocast.metrics.solvability import is_route_solved
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]


def check_solvability_with_stock(benchmark_path: Path, stock_path: Path) -> None:
    """
    Check how many acceptable routes are solvable with a given stock.

    Args:
        benchmark_path: Path to benchmark JSON.GZ file
        stock_path: Path to stock CSV.GZ file
    """
    logger.info(f"Loading benchmark from {benchmark_path}")
    benchmark = load_benchmark(benchmark_path)

    logger.info(f"Loading stock from {stock_path}")
    stock = load_stock_file(stock_path)
    logger.info(f"Stock contains {len(stock)} molecules")

    # Check solvability
    total_routes = sum(len(target.acceptable_routes) for target in benchmark.targets.values())
    solvable_routes = 0
    unsolvable_targets = []

    for target_id, target in benchmark.targets.items():
        for route_idx, route in enumerate(target.acceptable_routes):
            if is_route_solved(route, stock):
                solvable_routes += 1
            else:
                # Track which targets have unsolvable routes
                missing_leaves = [leaf.inchikey for leaf in route.leaves if leaf.inchikey not in stock]
                unsolvable_targets.append((target_id, route_idx, len(missing_leaves), len(route.leaves)))

    logger.info("\n=== Solvability Report ===")
    logger.info(f"Total targets: {len(benchmark.targets)}")
    logger.info(f"Total acceptable routes: {total_routes}")
    logger.info(f"Solvable routes: {solvable_routes}/{total_routes}")
    logger.info(f"Unsolvable routes: {len(unsolvable_targets)}/{total_routes}")

    if solvable_routes > 0:
        solvability_pct = (solvable_routes / total_routes) * 100
        logger.info(f"Solvability: {solvability_pct:.1f}%")

    if unsolvable_targets:
        logger.warning("\nFirst 5 unsolvable routes:")
        for target_id, route_idx, missing_count, total_leaves in unsolvable_targets[:5]:
            logger.warning(f"  {target_id}[{route_idx}]: {missing_count}/{total_leaves} leaves missing from stock")


if __name__ == "__main__":
    configure_script_logging()
    check_solvability_with_stock(
        benchmark_path=BASE_DIR / "data" / "1-benchmarks" / "definitions" / "uspto-190-retro-pickle.json.gz",
        stock_path=BASE_DIR / "data" / "1-benchmarks" / "stocks" / "buyables-stock.csv.gz",
    )
