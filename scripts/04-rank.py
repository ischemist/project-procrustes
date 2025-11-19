"""
Generates a probabilistic ranking of models using Monte Carlo simulation.

This script answers the question: "What is the probability that Model X is the best?"
It accounts for statistical uncertainty by simulating the ranking process 10,000 times
using the bootstrap distributions.

Usage:
    uv run scripts/04-rank.py --benchmark stratified-linear-600 --models dms-flash dms-wide dms-deep dms-flash-20M dms-explorer-xl dms-flex-duo
"""

import argparse
from pathlib import Path

from retrocast.io.files import load_json_gz
from retrocast.metrics.bootstrap import get_is_solvable, make_get_top_k
from retrocast.metrics.ranking import compute_probabilistic_ranking
from retrocast.models.evaluation import EvaluationResults
from retrocast.utils.logging import logger
from retrocast.visualization.model_performance import plot_probabilistic_ranking

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_eval_results(benchmark: str, model: str, stock: str) -> EvaluationResults | None:
    path = DATA_DIR / "4-scored" / benchmark / model / stock / "evaluation.json.gz"
    if not path.exists():
        logger.warning(f"Evaluation not found for {model}: {path}")
        return None
    return EvaluationResults.model_validate(load_json_gz(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate probabilistic model rankings.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to rank")
    parser.add_argument("--stock", default="n5-stock", help="Stock used for evaluation")
    parser.add_argument(
        "--metric",
        default="top-1",
        choices=["top-1", "top-5", "top-10", "solvability"],
        help="Metric to use for ranking",
    )
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading models for benchmark: {args.benchmark}")
    loaded_models: dict[str, EvaluationResults] = {}

    for model in args.models:
        res = load_eval_results(args.benchmark, model, args.stock)
        if res:
            loaded_models[model] = res

    if len(loaded_models) < 2:
        logger.error("Need at least 2 valid models to perform ranking.")
        return

    # 2. Determine Metric
    if args.metric == "solvability":
        extractor = get_is_solvable
        label = "Solvability"
    elif args.metric.startswith("top-"):
        k = int(args.metric.split("-")[1])
        extractor = make_get_top_k(k)
        label = f"Top-{k} Accuracy"
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    # 3. Compute Probabilistic Ranking
    logger.info(f"Ranking based on {label}...")
    ranking = compute_probabilistic_ranking(model_results=loaded_models, metric_extractor=extractor, n_boot=10000)

    # 4. Print Summary Table
    print(f"\n=== Ranking Results ({label}) ===")
    print(f"{'Model':<20} | {'Exp. Rank':<10} | {'Prob. Rank 1':<12}")
    print("-" * 46)

    for r in ranking:
        prob_first = r.rank_probs.get(1, 0.0)
        print(f"{r.model_name:<20} | {r.expected_rank:.2f}       | {prob_first:.1%}")
    print("-" * 46)

    # 5. Generate Plot
    output_dir = DATA_DIR / "6-comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_probabilistic_ranking(ranking, label)

    out_file = output_dir / f"ranking_heatmap_{args.metric}.html"
    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)

    logger.info(f"\nInteractive heatmap saved to: {out_file}")


if __name__ == "__main__":
    main()
