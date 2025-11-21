"""
Generates a probabilistic ranking of models using Monte Carlo simulation.

This script answers the question: "What is the probability that Model X is the best?"
It accounts for statistical uncertainty by simulating the ranking process 10,000 times
using the bootstrap distributions.

Usage:
    uv run scripts/04-rank.py --benchmark stratified-linear-600-seed=42 --models dms-flash dms-wide dms-deep dms-explorer-xl dms-flash-20M
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.metrics.bootstrap import get_is_solvable, make_get_top_k
from retrocast.metrics.ranking import compute_probabilistic_ranking
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import plots
from retrocast.visualization.report import create_ranking_table

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

console = Console()


def main() -> None:
    configure_script_logging(use_rich=True)
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
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap simulations")
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Data
    loaded_models = {}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task_load = progress.add_task("Loading models...", total=len(args.models))

        for model in args.models:
            res = loader.load_evaluation(args.benchmark, model, args.stock)
            if res:
                loaded_models[model] = res
            progress.advance(task_load)

    if len(loaded_models) < 2:
        logger.error("[bold red]Need at least 2 valid models to perform ranking.[/]")
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

    # 3. Compute Ranking
    logger.info(f"Simulating ranking (N={args.n_boot}) based on [bold]{label}[/]...")

    ranking = compute_probabilistic_ranking(model_results=loaded_models, metric_extractor=extractor, n_boot=args.n_boot)

    # 4. Display Table
    table = create_ranking_table(ranking, label)
    console.print(table)

    # 5. Generate Plot
    output_dir = DATA_DIR / "6-comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plots.plot_ranking(ranking, label)

    out_file = output_dir / f"ranking_heatmap_{args.metric}.html"
    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)

    logger.info(f"Interactive heatmap saved to: [underline]{out_file}[/]")


if __name__ == "__main__":
    main()
