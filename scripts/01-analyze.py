"""
Generates statistics and report for a scored model.

Usage:
    uv run scripts/01-analyze.py --model dms-flash-fp16 --stock n5-stock --benchmark random-n5-500
    uv run scripts/01-analyze.py --model dms-flash-fp16 --stock n5-stock --benchmark stratified-linear-600
    uv run scripts/01-analyze.py --model dms-flash-fp16 --stock n5-stock --benchmark stratified-convergent-250
"""

import argparse
from pathlib import Path

from retrocast.io.files import load_json_gz, save_json_gz
from retrocast.metrics.bootstrap import compute_metric_with_ci, get_is_solvable, make_get_top_k
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelStatistics
from retrocast.utils.logging import logger
from retrocast.visualization.model_performance import plot_single_model_diagnostics
from retrocast.visualization.report import generate_markdown_report

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--stock", required=True)
    args = parser.parse_args()

    # 1. Load Scored Results
    score_path = DATA_DIR / "4-scored" / args.benchmark / args.model / args.stock / "evaluation.json.gz"
    if not score_path.exists():
        logger.error(f"Scored results not found: {score_path}")
        return

    logger.info("Loading evaluation results...")
    # We use generic load because we need to cast it to Pydantic ourselves
    # (or update io/loaders.py to have load_evaluation_results)
    raw_data = load_json_gz(score_path)
    eval_results = EvaluationResults.model_validate(raw_data)

    targets = list(eval_results.results.values())

    # 2. Compute Stats
    logger.info("Bootstrapping statistics...")

    # Helper to group by length
    def get_length(t):
        return t.route_length

    # Solvability
    stat_solvability = compute_metric_with_ci(targets, get_is_solvable, "Solvability", group_by=get_length)

    # Top-K
    stat_topk = {}
    for k in [1, 5, 10]:
        stat_topk[k] = compute_metric_with_ci(targets, make_get_top_k(k), f"Top-{k}", group_by=get_length)

    # Assemble
    final_stats = ModelStatistics(
        model_name=args.model,
        benchmark=args.benchmark,
        stock=args.stock,
        solvability=stat_solvability,
        top_k_accuracy=stat_topk,
    )

    # 3. Save JSON
    output_dir = DATA_DIR / "5-results" / args.benchmark / args.model / args.stock
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json_gz(final_stats, output_dir / "statistics.json.gz")

    # 4. Generate Report
    report = generate_markdown_report(final_stats)
    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    logger.info(f"Analysis complete. Report saved to {output_dir / 'report.md'}")
    print("\n" + report)

    fig = plot_single_model_diagnostics(final_stats)
    fig.write_html(output_dir / "diagnostics.html", include_plotlyjs="cdn", auto_open=False)
    logger.info(f"Diagnostics plot saved to {output_dir / 'diagnostics.html'}")


if __name__ == "__main__":
    main()
