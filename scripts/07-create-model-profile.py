"""
Compares a single model across multiple benchmarks.

This script leverages the existing visualization infrastructure by performing a
'masquerade': it loads statistics for the same model across different benchmarks,
but temporarily renames the model to the benchmark name. This allows the
plotting adapters to treat benchmarks as 'competing models' for visualization.

Usage:
    uv run scripts/07-create-model-profile.py \
      --model dms-explorer-xl \
      --benchmarks \
        ref-lin-600:n5-stock \
        ref-cnv-400:n5-stock \
        ref-lng-84:n1-n5-stock \
        mkt-lin-500:buyables-stock \
        mkt-cnv-160:buyables-stock
"""

import argparse
import logging
from copy import deepcopy
from pathlib import Path

from rich.progress import track

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import plots

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def parse_benchmark_arg(arg: str, default_stock: str) -> tuple[str, str]:
    """Parses 'benchmark:stock' or returns default stock."""
    if ":" in arg:
        bench, stock = arg.split(":", 1)
        return bench, stock
    return arg, default_stock


def main():
    configure_script_logging(use_rich=True)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plots for a single model.")

    parser.add_argument("--model", required=True, help="Name of the model to analyze")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="List of benchmarks. Format: 'name' (uses default stock) or 'name:stock_name'",
    )
    parser.add_argument(
        "--default-stock", default="n5-stock", help="Default stock if not specified in benchmark string"
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 10, 20, 50],
        help="Top-K values to show in overall summary",
    )
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Data
    logger.info(
        f"Loading statistics for model [bold cyan]{args.model}[/] across [bold magenta]{len(args.benchmarks)}[/] benchmarks..."
    )

    stats_list = []

    # We loop over benchmarks, loading the SAME model for each
    for bench_arg in args.benchmarks:
        bench_name, stock_name = parse_benchmark_arg(bench_arg, args.default_stock)
        try:
            # Load the specific model for this benchmark
            # Returns a list, but we only asked for one model
            loaded_stats = loader.load_statistics(bench_name, [args.model], stock_name)

            if not loaded_stats:
                logger.warning(f"Model {args.model} not found for benchmark {bench_name}. Skipping.")
                continue

            # Get the stats object
            original_stats = loaded_stats[0]

            # THE MASQUERADE:
            # We create a copy and overwrite 'model_name' with 'benchmark'.
            # This tricks the plotting functions into grouping by Benchmark.
            stats_proxy = deepcopy(original_stats)
            stats_proxy.model_name = bench_name  # The plot legend will now show the benchmark name

            # Optional: Update the benchmark field to something generic to avoid confusion in titles
            stats_proxy.benchmark = "Benchmark Comparison"

            stats_list.append(stats_proxy)

        except Exception as e:
            logger.warning(f"Failed to load {bench_name}: {e}")

    if not stats_list:
        logger.error("[bold red]No valid statistics found. Exiting.[/]")
        return

    logger.info(f"Successfully prepared comparison for [green]{len(stats_list)}[/] benchmarks.")

    # 2. Prepare Output
    output_dir = DATA_DIR / "6-comparisons" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate Plots
    # We use the exact same plotting functions as the model comparison script.
    tasks = [
        (plots.plot_comparison, "compare_top1.html", {"models_stats": stats_list, "metric_type": "Top-K", "k": 1}),
        (plots.plot_comparison, "compare_solvability.html", {"models_stats": stats_list, "metric_type": "Solvability"}),
        (
            plots.plot_overall_summary,
            "compare_overall_summary.html",
            {"models_stats": stats_list, "top_k_values": args.top_k},
        ),
        # The matrix is particularly useful here: it shows the relative difficulty
        # of benchmark A vs benchmark B (e.g., "n5 is 10% harder than n1")
        (plots.plot_performance_matrix, "difficulty_matrix.html", {"models_stats": stats_list}),
    ]

    for plot_func, filename, kwargs in track(tasks, description="Generating plots..."):
        try:
            fig = plot_func(**kwargs)

            # Update title to reflect we are comparing benchmarks
            current_title = fig.layout.title.text
            if current_title:
                # Simple regex-like replace isn't perfect but works for the standard titles
                new_title = current_title.replace("Model Comparison", f"Benchmark Comparison ({args.model})")
                new_title = new_title.replace("Overall Performance Summary", f"Overall Performance ({args.model})")
                fig.update_layout(title=new_title)

            out_path = output_dir / filename
            fig.write_html(out_path, include_plotlyjs="cdn", auto_open=True)
        except Exception as e:
            logger.error(f"Failed to generate {filename}: {e}")

    logger.info(f"[bold green]Done![/] Reports saved to: [underline]{output_dir}[/]")


if __name__ == "__main__":
    main()
