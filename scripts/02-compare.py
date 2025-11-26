"""
Compares multiple scored models on the same benchmark.

Usage:
    uv run scripts/02-compare.py --benchmark uspto-190 --models dms-explorer-xl dms-flash dms-wide aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro askcos --stock buyables-stock

    uv run scripts/02-compare.py --benchmark mkt-cnv-160 --models dms-explorer-xl aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro --stock buyables-stock

    uv run scripts/02-compare.py --benchmark mkt-lin-500 --models dms-explorer-xl aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro --stock buyables-stock

    uv run scripts/02-compare.py --benchmark ref-lng-84 --models dms-explorer-xl aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro --stock n1-n5-stock

    uv run scripts/02-compare.py --benchmark ref-lin-600 --models dms-explorer-xl aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro --stock n5-stock

    uv run scripts/02-compare.py --benchmark ref-cnv-400 --models dms-explorer-xl aizynthfinder-mcts aizynthfinder-retro-star retro-star retro-star-high syntheseus-retro0-local-retro --stock n5-stock
"""

import argparse
import logging
from pathlib import Path

from rich.progress import track

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import plots

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main():
    configure_script_logging(use_rich=True)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plots.")
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark set")
    parser.add_argument("--stock", required=True, help="Stock definition used")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to compare")
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 10, 20, 50],
        help="Top-K values to show in overall summary (default: 1, 2, 3, 4, 5, 10, 20, 50)",
    )
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Data
    logger.info(f"Loading statistics for [bold cyan]{len(args.models)}[/] models...")

    # We use the loader to fetch valid stats objects
    stats_list = loader.load_statistics(args.benchmark, args.models, args.stock)

    if not stats_list:
        logger.error("[bold red]No valid statistics found. Exiting.[/]")
        return

    logger.info(f"Successfully loaded [green]{len(stats_list)}[/] models.")

    # 2. Prepare Output
    output_dir = DATA_DIR / "6-comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate Plots
    # We define the plot generation tasks as (Function, Filename, Kwargs)
    tasks = [
        (plots.plot_comparison, "compare_top1.html", {"models_stats": stats_list, "metric_type": "Top-K", "k": 1}),
        (plots.plot_comparison, "compare_solvability.html", {"models_stats": stats_list, "metric_type": "Solvability"}),
        (
            plots.plot_overall_summary,
            "compare_overall_summary.html",
            {"models_stats": stats_list, "top_k_values": args.top_k},
        ),
        (plots.plot_performance_matrix, "compare_matrix.html", {"models_stats": stats_list}),
    ]

    # Run with progress bar
    for plot_func, filename, kwargs in track(tasks, description="Generating plots..."):
        try:
            fig = plot_func(**kwargs)
            out_path = output_dir / filename
            fig.write_html(out_path, include_plotlyjs="cdn", auto_open=True)
        except Exception as e:
            logger.error(f"Failed to generate {filename}: {e}")

    logger.info(f"[bold green]Done![/] Reports saved to: [underline]{output_dir}[/]")


if __name__ == "__main__":
    main()
