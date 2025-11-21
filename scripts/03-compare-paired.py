"""
Runs paired statistical tests between a baseline model and challengers.
Calculates the difference in performance (Challenger - Baseline) with bootstrap CIs.

Usage:
    uv run scripts/03-compare-paired.py --benchmark stratified-linear-600-seed=42 --baseline dms-deep --challengers dms-flash dms-wide
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.metrics.bootstrap import (
    compute_paired_difference,
    get_is_solvable,
    make_get_top_k,
)
from retrocast.models.stats import ModelComparison
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization.report import create_paired_comparison_table

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

console = Console()


def main() -> None:
    configure_script_logging(use_rich=True)
    parser = argparse.ArgumentParser(description="Run paired difference tests.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--baseline", required=True, help="Model to compare against")
    parser.add_argument("--challengers", nargs="+", required=True, help="Models to compare")
    parser.add_argument("--stock", default="n5-stock", help="Stock used for evaluation")
    parser.add_argument("--n-boot", type=int, default=5000, help="Number of bootstrap samples")
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Baseline
    logger.info(f"Loading baseline: [bold cyan]{args.baseline}[/]")
    baseline_res = loader.load_evaluation(args.benchmark, args.baseline, args.stock)
    if not baseline_res:
        return

    # Extract lists of targets (ensuring order if IDs match, but EvalResults usually ensures this)
    # Ideally, we'd align by ID here, but assuming deterministic ordering for now based on load
    baseline_targets = list(baseline_res.results.values())

    # 2. Define Metrics
    metrics_config = [
        ("Solvability", get_is_solvable),
        ("Top-1", make_get_top_k(1)),
        ("Top-5", make_get_top_k(5)),
        ("Top-10", make_get_top_k(10)),
    ]

    all_comparisons: list[ModelComparison] = []

    # 3. Run Comparisons
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task_id = progress.add_task("Bootstrapping...", total=len(args.challengers))

        for challenger_name in args.challengers:
            progress.update(task_id, description=f"Comparing vs {challenger_name}...")

            challenger_res = loader.load_evaluation(args.benchmark, challenger_name, args.stock)
            if not challenger_res:
                continue

            challenger_targets = list(challenger_res.results.values())

            for metric_name, extractor in metrics_config:
                try:
                    comp = compute_paired_difference(
                        targets_a=baseline_targets,
                        targets_b=challenger_targets,
                        metric_extractor=extractor,
                        model_a_name=args.baseline,
                        model_b_name=challenger_name,
                        metric_name=metric_name,
                        n_boot=args.n_boot,
                    )
                    all_comparisons.append(comp)

                except ValueError as e:
                    logger.error(f"Error processing {metric_name} for {challenger_name}: {e}")

            progress.advance(task_id)

    # 4. Display Report
    if not all_comparisons:
        logger.warning("No comparisons generated.")
        return

    table = create_paired_comparison_table(
        baseline_name=args.baseline, benchmark_name=args.benchmark, comparisons=all_comparisons
    )

    console.print(table)
    console.print(
        "\n[dim]* Positive Diff = Challenger is better.\n"
        "* ✅ = 0 is not in the 95% Confidence Interval (Statistically Significant).[/]"
    )


if __name__ == "__main__":
    main()
