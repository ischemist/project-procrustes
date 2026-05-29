"""
run paired statistical tests between a baseline model and challengers.

usage:
    uv run scripts/03-compare-paired.py --benchmark small --baseline model-a --challengers model-b
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from retrocast.io import load_evaluation
from retrocast.metrics.ranking import PairwiseComparison, compute_paired_difference
from retrocast.models.evaluation import TargetResult, Tier
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data" / "retrocast"

console = Console()


def main() -> None:
    configure_script_logging(use_rich=True)
    parser = argparse.ArgumentParser(description="run paired difference tests over schema v2 evaluations.")
    parser.add_argument("--benchmark", required=True, help="benchmark name")
    parser.add_argument("--baseline", required=True, help="model to compare against")
    parser.add_argument("--challengers", nargs="+", required=True, help="models to compare")
    parser.add_argument("--stock", help="stock/result label used under 4-scored; inferred when only one exists")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-boot", type=int, default=5000, help="number of bootstrap samples")
    args = parser.parse_args()

    stock = args.stock or infer_stock_label(args.data_dir, args.benchmark, args.baseline)
    baseline_targets = load_targets(args.data_dir, args.benchmark, args.baseline, stock)
    metrics: list[tuple[str, Callable[[TargetResult], float]]] = [
        ("Solv-0", solv_metric(Tier.ZERO)),
        ("Top-1", top_k_metric(1)),
        ("Top-5", top_k_metric(5)),
        ("Top-10", top_k_metric(10)),
    ]

    comparisons = []
    for challenger in args.challengers:
        challenger_targets = load_targets(args.data_dir, args.benchmark, challenger, stock)
        for metric_name, extractor in metrics:
            comparisons.append(
                compute_paired_difference(
                    baseline_targets,
                    challenger_targets,
                    extractor,
                    model_a_name=args.baseline,
                    model_b_name=challenger,
                    metric_name=metric_name,
                    n_boot=args.n_boot,
                )
            )

    if not comparisons:
        logger.warning("no comparisons generated")
        return
    console.print(create_paired_comparison_table(args.baseline, args.benchmark, comparisons))


def load_targets(data_dir: Path, benchmark: str, model: str, stock: str) -> list[TargetResult]:
    path = data_dir / "4-scored" / benchmark / model / stock / "evaluation.json.gz"
    return list(load_evaluation(path).targets.values())


def infer_stock_label(data_dir: Path, benchmark: str, model: str) -> str:
    scored_dir = data_dir / "4-scored" / benchmark / model
    if not scored_dir.exists():
        raise FileNotFoundError(f"scored directory not found: {scored_dir}")
    labels = sorted(path.name for path in scored_dir.iterdir() if path.is_dir())
    if len(labels) != 1:
        raise ValueError(f"--stock is required when {scored_dir} contains {len(labels)} result labels")
    return labels[0]


def solv_metric(tier: Tier) -> Callable[[TargetResult], float]:
    def extract(target: TargetResult) -> float:
        return 1.0 if any(candidate.satisfies_solv(tier) for candidate in target.candidates) else 0.0

    return extract


def top_k_metric(k: int) -> Callable[[TargetResult], float]:
    def extract(target: TargetResult) -> float:
        ranked = sorted(target.candidates, key=lambda candidate: candidate.rank)
        satisfying = [candidate for candidate in ranked if candidate.satisfies_task()]
        return 1.0 if any(candidate.matches_acceptable for candidate in satisfying[:k]) else 0.0

    return extract


def create_paired_comparison_table(
    baseline_name: str,
    benchmark_name: str,
    comparisons: list[PairwiseComparison],
) -> Table:
    table = Table(title=f"paired comparisons vs {baseline_name}: {benchmark_name}")
    table.add_column("challenger")
    table.add_column("metric")
    table.add_column("diff", justify="right")
    table.add_column("95% ci", justify="center")
    table.add_column("sig", justify="center")
    for comparison in comparisons:
        table.add_row(
            comparison.model_b,
            comparison.metric,
            f"{comparison.diff_mean:+.3f}",
            f"[{comparison.diff_ci_low:+.3f}, {comparison.diff_ci_high:+.3f}]",
            "yes" if comparison.is_significant else "no",
        )
    return table


if __name__ == "__main__":
    main()
