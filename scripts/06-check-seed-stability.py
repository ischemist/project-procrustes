"""
Analyzes the stability of model performance across different benchmark seeds.
Generates a Forest Plot showing variance due to subset sampling.

Usage:
    uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark stratified-linear-600 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716

    uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark stratified-convergent-450 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from ischemist.plotly import Styler

from retrocast.io.files import load_json_gz
from retrocast.metrics.bootstrap import compute_metric_with_ci, get_is_solvable, make_get_top_k
from retrocast.models.evaluation import EvaluationResults
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_scored_data(benchmark_name: str, model: str, stock: str) -> list | None:
    """Loads scored targets for a specific benchmark variant."""
    path = DATA_DIR / "4-scored" / benchmark_name / model / stock / "evaluation.json.gz"
    if not path.exists():
        logger.warning(f"Missing data for {benchmark_name}")
        return None
    return list(EvaluationResults.model_validate(load_json_gz(path)).results.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-benchmark", required=True, help="Base name (e.g. stratified-linear-600)")
    parser.add_argument("--seeds", nargs="+", required=True, help="List of seeds to check")
    parser.add_argument("--stock", default="n5-stock")
    args = parser.parse_args()

    results_map = {}  # seed -> {metric: MetricResult}

    # 1. Compute Stats for Each Seed
    for seed in args.seeds:
        bench_name = f"{args.base_benchmark}-seed={seed}"
        logger.info(f"Processing {bench_name}...")

        targets = load_scored_data(bench_name, args.model, args.stock)
        if not targets:
            continue

        # Calculate Solvability
        res_solv = compute_metric_with_ci(targets, get_is_solvable, "Solvability")
        # Calculate Top-1
        res_top1 = compute_metric_with_ci(targets, make_get_top_k(1), "Top-1")
        res_top10 = compute_metric_with_ci(targets, make_get_top_k(10), "Top-10")

        results_map[seed] = {"Solvability": res_solv.overall, "Top-1": res_top1.overall, "Top-10": res_top10.overall}

    if not results_map:
        logger.error("No valid data found.")
        return

    # 2. Plotting Logic
    # We create two subplots (or two traces on one plot if scales allow).
    # Since scales are 0-100, we can put them on one plot with color grouping.

    fig = go.Figure()

    seeds_sorted = sorted(results_map.keys(), key=lambda x: int(x))

    # Metrics to plot
    metrics = ["Solvability", "Top-1", "Top-10"]
    colors = {"Solvability": "#2E86C1", "Top-1": "#E67E22", "Top-10": "#9B59B6"}

    for metric in metrics:
        x_vals = []
        y_vals = []  # Seed names
        error_plus = []
        error_minus = []

        raw_values = []  # For calculating grand mean

        for seed in seeds_sorted:
            res = results_map[seed][metric]

            x_vals.append(res.value * 100)
            y_vals.append(f"Seed {seed}")

            error_plus.append((res.ci_upper - res.value) * 100)
            error_minus.append((res.value - res.ci_lower) * 100)

            raw_values.append(res.value * 100)

        # Add Trace
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                name=metric,
                mode="markers",
                marker=dict(color=colors[metric], size=10),
                error_x=dict(type="data", array=error_plus, arrayminus=error_minus, visible=True),
            )
        )

        # Add Grand Mean Line
        grand_mean = np.mean(raw_values)
        std_dev = np.std(raw_values)

        fig.add_vline(
            x=grand_mean,
            line_width=1,
            line_dash="dash",
            line_color=colors[metric],
            annotation_text=f"Mean: {grand_mean:.1f}% (σ={std_dev:.2f})",
            annotation_position="top right",
        )

    # 3. Layout Polish
    fig.update_layout(
        title=f"Benchmark Stability: {args.base_benchmark}",
        xaxis_title="Performance (%)",
        yaxis_title="Seed Variant",
        height=600,
    )
    Styler().apply_style(fig)

    out_dir = DATA_DIR / "7-meta-analysis" / args.base_benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "seed_stability.html"
    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)
    logger.info(f"Stability plot saved to {out_file}")

    # Log the variance stats to console
    print("\n=== Stability Summary ===")
    metric_stats = {}
    for metric in metrics:
        vals = [results_map[s][metric].value * 100 for s in seeds_sorted]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        metric_stats[metric] = {"mean": mean_val, "std": std_val}
        print(f"{metric}: Mean={mean_val:.2f}%, StdDev={std_val:.2f}%")

    # Calculate and print deviation scores (z-score sum) for each seed
    print("\n=== Deviation Scores (Sum of Squared Z-Scores) ===")
    seed_deviations = []
    for seed in seeds_sorted:
        top1_val = results_map[seed]["Top-1"].value * 100
        solv_val = results_map[seed]["Solvability"].value * 100
        top10_val = results_map[seed]["Top-10"].value * 100

        # Calculate z-scores
        z_top1 = (
            (top1_val - metric_stats["Top-1"]["mean"]) / metric_stats["Top-1"]["std"]
            if metric_stats["Top-1"]["std"] > 0
            else 0
        )
        z_solv = (
            (solv_val - metric_stats["Solvability"]["mean"]) / metric_stats["Solvability"]["std"]
            if metric_stats["Solvability"]["std"] > 0
            else 0
        )
        z_top10 = (
            (top10_val - metric_stats["Top-10"]["mean"]) / metric_stats["Top-10"]["std"]
            if metric_stats["Top-10"]["std"] > 0
            else 0
        )

        # Deviation score is sum of squared z-scores
        deviation_score = z_top1**2 + z_solv**2 + z_top10**2

        seed_deviations.append((seed, deviation_score, z_top1, z_solv, z_top10))
        print(
            f"Seed {seed}: Deviation={deviation_score:.4f} (z_top1={z_top1:+.3f}, z_solv={z_solv:+.3f}, z_top10={z_top10:+.3f})"
        )

    # Sort by deviation score and show most stable seeds
    seed_deviations.sort(key=lambda x: x[1])
    print("\n=== Most Stable Seeds (Lowest Deviation) ===")
    for i, (seed, dev) in enumerate(seed_deviations[:2], 1):
        print(f"{i}. Seed {seed}: Deviation={dev:.4f}")


if __name__ == "__main__":
    main()
