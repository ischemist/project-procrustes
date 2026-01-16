"""
Generates a Pareto frontier plot showing cost vs accuracy trade-offs.

This script creates a scatter plot with:
- X-axis: Total cost (computed from wall_time * hourly_cost)
- Y-axis: Top-K accuracy (default k=10)
- Each model as a point with error bars and labels

Usage:
    uv run scripts/08-pareto-frontier.py \
      --benchmark mkt-cnv-160 \
      --stock buyables-stock \
      --top-k 10

    uv run scripts/08-pareto-frontier.py \
      --benchmark mkt-cnv-160 \
      --stock buyables-stock \
      --models dms-explorer-xl aizynthfinder-mcts retro-star
"""

import argparse
import logging
from pathlib import Path

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import plots

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Hourly compute costs in USD
HOURLY_COSTS = {
    "aizynthfinder-retro-star": 0.1785,
    "aizynthfinder-mcts": 0.1785,
    # "retro-star": 0.1785,
    "syntheseus-retro0-local-retro": 0.1785,
    "dms-explorer-xl": 1.29,
    "aizynthfinder-retro-star-high": 0.1785,
    "aizynthfinder-mcts-high": 0.1785,
    "retro-star-high": 0.1785,
    "askcos": 0.714,
    # "synplanner-mcts": 0.1785,
    # "synplanner-eval": 0.1785,
}

# Model display configuration
MODEL_CONFIG = {
    "dms-explorer-xl": {
        "legend": "DMS Explorer XL",
        "short": "DMS XL",
        "color": "#5eaff2",
    },
    "aizynthfinder-mcts": {
        "legend": "AiZynthFinder MCTS",
        "short": "AZF MCTS",
        "color": "#f7b267",
    },
    "aizynthfinder-mcts-high": {
        "legend": "AiZynthFinder MCTS (High)",
        "short": "AZF MCTS H",
        "color": "#f79d65",
    },
    "aizynthfinder-retro-star": {
        "legend": "AiZynthFinder RetroStar",
        "short": "AZF RS",
        "color": "#f4845f",
    },
    "aizynthfinder-retro-star-high": {
        "legend": "AiZynthFinder RetroStar (High)",
        "short": "AZF RS H",
        "color": "#f27059",
    },
    "retro-star": {
        "legend": "RetroStar",
        "short": "RS",
        "color": "#d4abef",
    },
    "retro-star-high": {
        "legend": "RetroStar (High)",
        "short": "RS H",
        "color": "#9f85ff",
    },
    "askcos": {
        "legend": "ASKCOS",
        "short": "ASKCOS",
        "color": "#fe7295",
    },
    "syntheseus-retro0-local-retro": {
        "legend": "Syntheseus Retro0",
        "short": "Synth R0",
        "color": "#d6d2d2",
    },
    # "synplanner-mcts": {
    #     "legend": "SynPlanner MCTS",
    #     "short": "SP MCTS",
    #     "color": "#e377c2",  # Pink
    # },
    # "synplanner-eval": {
    #     "legend": "SynPlanner Eval",
    #     "short": "SP Eval",
    #     "color": "#f06292",  # Lighter pink
    # },
}


def main():
    configure_script_logging(use_rich=True)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot (cost vs accuracy).")
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark set")
    parser.add_argument("--stock", required=True, help="Stock definition used")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to include (default: all models with cost data)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Which top-k accuracy to plot on Y-axis (default: 10)",
    )
    parser.add_argument(
        "--time-based",
        action="store_true",
        help="Use wall time (minutes) instead of cost (USD) for X-axis",
    )
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Data
    logger.info(f"Loading statistics for benchmark [bold cyan]{args.benchmark}[/]...")

    # If no models specified, try to load all models with known costs
    if args.models:
        model_list = args.models
    else:
        model_list = list(HOURLY_COSTS.keys())
        logger.info(f"No models specified. Attempting to load all {len(model_list)} models with cost data...")

    stats_list = loader.load_statistics(args.benchmark, model_list, args.stock)

    if not stats_list:
        logger.error("[bold red]No valid statistics found. Exiting.[/]")
        return

    logger.info(f"Successfully loaded [green]{len(stats_list)}[/] models.")

    # 2. Prepare Output
    output_dir = DATA_DIR / "6-comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate Plot
    logger.info("Generating Pareto frontier plot...")

    try:
        fig = plots.plot_pareto_frontier(
            models_stats=stats_list,
            model_config=MODEL_CONFIG,
            hourly_costs=HOURLY_COSTS,
            k=args.top_k,
            time_based=args.time_based,
        )

        suffix = "_time" if args.time_based else ""
        out_path = output_dir / f"pareto_frontier{suffix}.html"
        fig.write_html(out_path, include_plotlyjs="cdn", auto_open=True)
        fig.write_image(output_dir / f"{args.benchmark}{suffix}.pdf")
        logger.info(f"[bold green]Done![/] Plot saved to: [underline]{out_path}[/]")

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        raise


if __name__ == "__main__":
    main()
