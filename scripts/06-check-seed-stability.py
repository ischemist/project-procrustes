"""
Analyzes the stability of model performance across different benchmark seeds.
Generates a Forest Plot showing variance due to subset sampling.

Usage:
    uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark stratified-linear-600 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import track

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.metrics.bootstrap import compute_metric_with_ci, get_is_solvable, make_get_top_k
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import adapters, plots, theme
from retrocast.visualization.report import create_stability_table

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

console = Console()


def main():
    configure_script_logging(use_rich=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-benchmark", required=True, help="Base name (e.g. stratified-linear-600)")
    parser.add_argument("--seeds", nargs="+", required=True, help="List of seeds to check")
    parser.add_argument("--stock", required=True, help="Stock to use for evaluation")
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)
    results_map = {}  # seed -> {metric: MetricResult}

    # 1. Compute Stats for Each Seed
    for seed in track(args.seeds, description="Analyzing seeds..."):
        bench_name = f"{args.base_benchmark}-seed={seed}"

        # We load raw targets because we want to re-compute stats on the fly
        # (or we could load pre-computed stats if available, but raw is safer for ad-hoc analysis)
        eval_res = loader.load_evaluation(bench_name, args.model, args.stock)

        if not eval_res:
            continue

        targets = list(eval_res.results.values())

        # Calculate Metrics
        res_solv = compute_metric_with_ci(targets, get_is_solvable, "Solvability")
        res_top1 = compute_metric_with_ci(targets, make_get_top_k(1), "Top-1")
        res_top10 = compute_metric_with_ci(targets, make_get_top_k(10), "Top-10")

        results_map[seed] = {"Solvability": res_solv.overall, "Top-1": res_top1.overall, "Top-10": res_top10.overall}

    if not results_map:
        logger.error("[bold red]No valid data found for any seed.[/]")
        return

    # 2. Prepare Plot Data
    plot_data = []
    metrics_summary = {}  # For the table

    # Define metrics to analyze
    configs = [("Solvability", theme.COLOR_SOLVABILITY), ("Top-1", theme.COLOR_TOP_1), ("Top-10", theme.COLOR_TOP_10)]

    for metric_key, color in configs:
        # Create Adapter for Plot
        s_data = adapters.stats_to_stability_data(results_map, metric_key, color)
        plot_data.append(s_data)

        # Store summary for text output
        metrics_summary[metric_key] = {"mean": s_data.grand_mean, "std": s_data.std_dev}

    # 3. Calculate Deviation Scores (Which seed is "most normal"?)
    seed_deviations = []
    seeds_sorted = sorted(results_map.keys(), key=lambda x: int(x) if x.isdigit() else x)

    def z(val: float, key: str) -> float:
        return (
            (val - metrics_summary[key]["mean"]) / metrics_summary[key]["std"] if metrics_summary[key]["std"] > 0 else 0
        )

    for seed in seeds_sorted:
        # Extract percent values
        v_top1 = results_map[seed]["Top-1"].value * 100
        v_solv = results_map[seed]["Solvability"].value * 100
        v_top10 = results_map[seed]["Top-10"].value * 100

        # Z-Scores: (Val - Mean) / Std

        z1 = z(v_top1, "Top-1")
        zs = z(v_solv, "Solvability")
        z10 = z(v_top10, "Top-10")

        # Score = Sum of Squared Z-Scores (lower is better/more typical)
        score = z1**2 + zs**2 + z10**2
        seed_deviations.append((seed, score, z1, zs, z10))

    # Sort by score
    seed_deviations.sort(key=lambda x: x[1])

    # 4. Render Output
    t1, t2 = create_stability_table(metrics_summary, seed_deviations)
    console.print(t1)
    console.print(t2)

    out_dir = DATA_DIR / "7-meta-analysis" / args.base_benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plots.plot_stability_analysis(plot_data, args.base_benchmark, args.model)
    out_file = out_dir / "seed_stability.html"

    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)
    fig.write_image(out_dir / "seed_stability.jpg", scale=4, height=600, width=1200)
    logger.info(f"Stability plot saved to: [underline]{out_file}[/]")

    # Suggest the best seed
    best_seed = seed_deviations[0][0]
    console.print(
        f"\n[bold green]Recommendation:[/] Use seed [bold cyan]{best_seed}[/] as the canonical representative."
    )


if __name__ == "__main__":
    main()
