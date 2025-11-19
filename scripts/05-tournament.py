"""
Runs a pairwise statistical tournament.
Generates a matrix showing the difference between every pair of models.

Usage:
    uv run scripts/05-tournament.py --benchmark stratified-linear-600 --models dms-flash dms-wide dms-deep dms-flash-20M dms-explorer-xl dms-flex-duo
"""

import argparse
from pathlib import Path

from retrocast.io.files import load_json_gz
from retrocast.metrics.bootstrap import get_is_solvable, make_get_top_k
from retrocast.metrics.ranking import compute_pairwise_tournament
from retrocast.models.evaluation import EvaluationResults
from retrocast.utils.logging import logger
from retrocast.visualization.model_performance import plot_pairwise_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_eval_results(benchmark: str, model: str, stock: str) -> EvaluationResults | None:
    path = DATA_DIR / "4-scored" / benchmark / model / stock / "evaluation.json.gz"
    if not path.exists():
        logger.warning(f"Evaluation not found: {path}")
        return None
    return EvaluationResults.model_validate(load_json_gz(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--stock", default="n5-stock")
    parser.add_argument("--metric", default="top-1", choices=["top-1", "solvability"])
    args = parser.parse_args()

    # 1. Load Data
    loaded_models = {}
    for model in args.models:
        res = load_eval_results(args.benchmark, model, args.stock)
        if res:
            loaded_models[model] = res

    if len(loaded_models) < 2:
        logger.error("Need 2+ models.")
        return

    # 2. Config
    if args.metric == "solvability":
        extractor = get_is_solvable
        label = "Solvability"
    else:
        extractor = make_get_top_k(1)
        label = "Top-1 Accuracy"

    # 3. Run Tournament
    logger.info("Running tournament...")
    results = compute_pairwise_tournament(loaded_models, extractor, label, n_boot=10000)

    # 4. Plot
    fig = plot_pairwise_matrix(results, label)

    out_dir = DATA_DIR / "6-comparisons" / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pairwise_matrix_{args.metric}.html"

    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)
    logger.info(f"Matrix saved to {out_file}")
    print("\nSaved. Look for the ★ in the plot. That means it's real.")


if __name__ == "__main__":
    main()
