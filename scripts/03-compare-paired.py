"""
Runs paired statistical tests between a baseline model and challengers.
Calculates the difference in performance (Baseline - Challenger) with bootstrap CIs.

Usage:
    uv run scripts/03-compare-paired.py --benchmark stratified-linear-600 --baseline dms-deep --challengers dms-flash-20M dms-wide
"""

import argparse
from pathlib import Path

from retrocast.io.files import load_json_gz
from retrocast.metrics.bootstrap import (
    compute_paired_difference,
    get_is_solvable,
    make_get_top_k,
)
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelComparison
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_eval_results(benchmark: str, model: str, stock: str) -> EvaluationResults | None:
    path = DATA_DIR / "4-scored" / benchmark / model / stock / "evaluation.json.gz"
    if not path.exists():
        logger.error(f"Evaluation not found for {model}: {path}")
        return None
    return EvaluationResults.model_validate(load_json_gz(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired difference tests.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--baseline", required=True, help="Model to compare against")
    parser.add_argument("--challengers", nargs="+", required=True, help="Models to compare")
    parser.add_argument("--stock", default="n5-stock", help="Stock used for evaluation")
    args = parser.parse_args()

    # 1. Load Baseline
    logger.info(f"Loading baseline: {args.baseline}")
    baseline_res = load_eval_results(args.benchmark, args.baseline, args.stock)
    if not baseline_res:
        return

    baseline_targets = list(baseline_res.results.values())

    all_comparisons: list[ModelComparison] = []

    # 2. Define Metrics to Test
    # (Metric Name, Extraction Function)
    metrics_config = [
        ("Solvability", get_is_solvable),
        ("Top-1", make_get_top_k(1)),
        ("Top-5", make_get_top_k(5)),
        ("Top-10", make_get_top_k(10)),
    ]

    # 3. Loop over Challengers
    for challenger_name in args.challengers:
        logger.info(f"Comparing vs {challenger_name}...")
        challenger_res = load_eval_results(args.benchmark, challenger_name, args.stock)

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
                )
                all_comparisons.append(comp)
            except ValueError as e:
                logger.error(f"Skipping {metric_name} for {challenger_name}: {e}")

    # 4. Print Report
    print(f"\n=== Paired Comparison (Baseline: {args.baseline}) ===")
    print(f"Benchmark: {args.benchmark} | Stock: {args.stock}")
    print("-" * 90)
    # Header
    print(f"| {'Challenger':<20} | {'Metric':<12} | {'Diff (Chal-Base)':<16} | {'95% CI':<18} | {'Sig?':<6} |")
    print("|" + "-" * 22 + "|" + "-" * 14 + "|" + "-" * 18 + "|" + "-" * 20 + "|" + "-" * 8 + "|")

    for c in all_comparisons:
        # Visual formatting
        sig_icon = "✅" if c.is_significant else "  "

        # Colorize difference?
        # If baseline is better (positive diff), that's usually "good" for the baseline.
        diff_str = f"{c.diff_mean:+.1%}"
        ci_str = f"[{c.diff_ci_lower:+.1%}, {c.diff_ci_upper:+.1%}]"

        print(f"| {c.model_b:<20} | {c.metric:<12} | {diff_str:<16} | {ci_str:<18} | {sig_icon:<5} |")

    print("-" * 90)
    print("* Positive Diff = Challenger is better. ✅ = 0 not in CI (Statistically Significant).")
    print("\n")


if __name__ == "__main__":
    main()
