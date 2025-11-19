"""
Compares multiple scored models on the same benchmark.

Usage:
    uv run scripts/02-compare.py --benchmark stratified-linear-600 --models dms-flash dms-flash-20M dms-flex-duo dms-wide dms-deep dms-explorer-xl
"""

import argparse
from pathlib import Path

from retrocast.io.files import load_json_gz
from retrocast.models.stats import ModelStatistics
from retrocast.utils.logging import logger
from retrocast.visualization.plots import plot_multi_model_comparison, plot_overall_comparison

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--stock", default="n5-stock")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names to compare")
    args = parser.parse_args()

    loaded_stats = []

    for model in args.models:
        # Path: data/results/{benchmark}/{model}/{stock}/statistics.json.gz
        stat_path = DATA_DIR / "5-results" / args.benchmark / model / args.stock / "statistics.json.gz"

        if not stat_path.exists():
            logger.warning(f"Stats not found for {model}: {stat_path}")
            continue

        raw = load_json_gz(stat_path)
        stats = ModelStatistics.model_validate(raw)
        loaded_stats.append(stats)

    if not loaded_stats:
        logger.error("No valid statistics loaded.")
        return

    # Generate Comparison Plots
    output_dir = DATA_DIR / "comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top-1 Comparison
    fig_top1 = plot_multi_model_comparison(loaded_stats, metric_type="Top-K", k=1)
    fig_top1.write_html(output_dir / "compare_top1.html", include_plotlyjs="cdn", auto_open=True)

    # 2. Solvability Comparison
    fig_solv = plot_multi_model_comparison(loaded_stats, metric_type="Solvability")
    fig_solv.write_html(output_dir / "compare_solvability.html", include_plotlyjs="cdn", auto_open=True)

    # 3. Overall Summary Plot
    fig_summary = plot_overall_comparison(loaded_stats)
    fig_summary.write_html(output_dir / "compare_overall.html", include_plotlyjs="cdn", auto_open=True)

    logger.info(f"Comparison plots saved to {output_dir}")


if __name__ == "__main__":
    main()
