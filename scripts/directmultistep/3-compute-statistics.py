"""
Compute aggregate benchmark statistics from scored evaluation results.

This script takes evaluation results (from 3-score-predictions.py) and computes
dataset-level statistics including:
- Overall solve rate and top-K accuracy
- Metrics stratified by experimental route length
- Sample sizes for transparency

Input:  data/scored/{model}/{dataset}/evaluation.json.gz
Output: data/statistics/{model}/{dataset}/{stock}/statistics.json
        data/statistics/{model}/{dataset}/{stock}/statistics.md

Usage:
    uv run scripts/directmultistep/3-compute-statistics.py
"""

import json
from pathlib import Path

from retrocast.evaluation.statistics import compute_benchmark_statistics, format_statistics_as_markdown
from retrocast.io import load_evaluation_results
from retrocast.utils.logging import logger

base_dir = Path(__file__).resolve().parents[2]
SCORED_DIR = base_dir / "data" / "scored"
STATISTICS_DIR = base_dir / "data" / "statistics"


def save_statistics_json(stats, output_path: Path) -> None:
    """Save statistics as JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    logger.info(f"Saved statistics to {output_path}")


def save_statistics_markdown(stats, output_path: Path) -> None:
    """Save statistics as markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = format_statistics_as_markdown(stats)
    output_path.write_text(markdown)
    logger.info(f"Saved markdown table to {output_path}")


def main():
    """Compute statistics for all available evaluation results."""
    model_name = "dms-flash-fp16"

    for dataset in ["n5"]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Computing statistics for {model_name} on {dataset}")
        logger.info(f"{'=' * 60}")

        # Load evaluation results
        eval_path = SCORED_DIR / model_name / dataset / "evaluation.json.gz"
        if not eval_path.exists():
            logger.warning(f"Evaluation file not found: {eval_path}")
            continue

        results = load_evaluation_results(eval_path)

        # Compute statistics for each stock
        for stock_name in results.stock_names:
            logger.info(f"\nProcessing stock: {stock_name}")

            stats = compute_benchmark_statistics(results, stock_name)

            # Save outputs
            output_dir = STATISTICS_DIR / model_name / dataset / stock_name
            save_statistics_json(stats, output_dir / "statistics.json")
            save_statistics_markdown(stats, output_dir / "statistics.md")

            logger.info(f"Completed {stock_name}")

        logger.info(f"\nCompleted {dataset}")


if __name__ == "__main__":
    main()
