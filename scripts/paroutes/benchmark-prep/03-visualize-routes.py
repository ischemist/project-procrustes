"""
Visualize n1 and n5 route statistics.

Loads processed n1 and n5 routes and creates a comparison figure with:
- Route count by depth
- Target heavy atom count by depth
- Target molecular weight by depth

Usage:
    uv run scripts/paroutes/benchmark-prep/03-visualize-routes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrocast.io import load_benchmark
from retrocast.utils.logging import logger
from retrocast.visualization.routes import create_route_comparison_figure, extract_route_stats

BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "retrocast" / "1-benchmarks" / "definitions"
OUTPUT_DIR = BASE_DIR / "data" / "retrocast" / "5-results" / "paroutes"


def main() -> None:
    """Main script execution."""
    args = parse_args()
    variants = ["", "-buyables"]
    for var in variants:
        logger.info("Loading routes...")
        n1_set = load_benchmark(PROCESSED_DIR / f"paroutes-n1-full{var}-pruned.json.gz")
        n5_set = load_benchmark(PROCESSED_DIR / f"paroutes-n5-full{var}-pruned.json.gz")

        n1_routes = {
            target_id: target.acceptable_routes[0]
            for target_id, target in n1_set.targets.items()
            if target.acceptable_routes
        }
        n5_routes = {
            target_id: target.acceptable_routes[0]
            for target_id, target in n5_set.targets.items()
            if target.acceptable_routes
        }

        logger.info("Extracting route statistics...")
        n1_stats = extract_route_stats(n1_routes)
        n5_stats = extract_route_stats(n5_routes)
        logger.info(f"n1: {len(n1_stats)} routes, n5: {len(n5_stats)} routes")

        logger.info("Creating figure...")
        fig = create_route_comparison_figure(n1_stats, n5_stats)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"route-comparison{var}.html"
        fig.write_html(output_file, include_plotlyjs="cdn", auto_open=not args.no_open)
        fig.write_image(output_file.with_suffix(".jpg"), scale=4, width=1200, height=1000)
        fig.write_image(output_file.with_suffix(".pdf"), width=1200, height=1000)
        logger.info(f"Saved figure to {output_file.relative_to(BASE_DIR)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PaRoutes n1/n5 route statistics.")
    parser.add_argument("--no-open", action="store_true", help="Write HTML without opening a browser.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
