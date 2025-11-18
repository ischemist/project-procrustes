"""
Visualize n1 and n5 route statistics.

Loads processed n1 and n5 routes and creates a comparison figure with:
- Route count by depth
- Target heavy atom count by depth
- Target molecular weight by depth

Usage:
    uv run scripts/paroutes/3-visualize-routes.py
"""

from pathlib import Path

from retrocast.analysis.routes import create_route_comparison_figure, extract_route_stats
from retrocast.io import load_routes
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "paroutes" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "analysis" / "paroutes"


def main() -> None:
    """Main script execution."""

    logger.info("Loading routes...")
    n1_routes = load_routes(PROCESSED_DIR / "n1-routes.json.gz")
    n5_routes = load_routes(PROCESSED_DIR / "n5-routes.json.gz")

    logger.info("Extracting route statistics...")
    n1_stats = extract_route_stats(n1_routes)
    n5_stats = extract_route_stats(n5_routes)
    logger.info(f"n1: {len(n1_stats)} routes, n5: {len(n5_stats)} routes")

    logger.info("Creating figure...")
    fig = create_route_comparison_figure(n1_stats, n5_stats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "route-comparison.html"
    fig.write_html(output_file, include_plotlyjs="cdn", auto_open=True)
    fig.write_image(output_file.with_suffix(".jpg"), scale=4, width=1200, height=1000)
    logger.info(f"Saved figure to {output_file.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
