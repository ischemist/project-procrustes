"""
retrocast.api

The public interface for using retrocast as a library.
Use these functions to run scoring and analysis programmatically
without relying on the specific directory structure of the CLI.
"""

from pathlib import Path

from retrocast.chem import InchiKeyLevel
from retrocast.io import load_benchmark, load_routes, load_stock_file
from retrocast.metrics.bootstrap import compute_metric_with_ci
from retrocast.metrics.ranking import compute_probabilistic_ranking
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelStatistics
from retrocast.typing import InchiKeyStr
from retrocast.workflow import analyze as analyze_workflow  # Added import
from retrocast.workflow import score as score_workflow

__all__ = [
    "load_benchmark",
    "load_routes",
    "load_stock_file",
    "score_predictions",
    "compute_model_statistics",
    "compute_metric_with_ci",
    "compute_probabilistic_ranking",
    "InchiKeyLevel",
]


def score_predictions(
    benchmark: BenchmarkSet,
    predictions: dict,
    stock: set[InchiKeyStr] | Path | str,
    model_name: str = "custom-model",
    stock_name: str | None = None,
    match_level: InchiKeyLevel = InchiKeyLevel.FULL,
) -> EvaluationResults:
    """
    Score a set of predictions against a benchmark and stock.

    Args:
        benchmark: The BenchmarkSet object.
        predictions: Dictionary mapping target_id -> list[Route].
        stock: Either a set of SMILES strings, or a Path to a stock file.
        model_name: Name to assign to these results.
        stock_name: Label for the stock. If None, inferred from Path or set to 'custom'.
        match_level: Level of InChI key matching specificity:
            - None or FULL: Exact matching (default)
            - NO_STEREO: Ignore stereochemistry
            - CONNECTIVITY: Match on molecular skeleton only
    """
    # Normalize stock input
    if isinstance(stock, (str, Path)):
        path = Path(stock)
        stock_set = load_stock_file(path)
        name = stock_name or path.stem
    else:
        stock_set = stock
        name = stock_name or "custom-stock"

    return score_workflow.score_model(
        benchmark=benchmark,
        predictions=predictions,
        stock=stock_set,
        stock_name=name,
        model_name=model_name,
        match_level=match_level,
    )


def compute_model_statistics(eval_results: EvaluationResults, n_boot: int = 10000, seed: int = 42) -> ModelStatistics:
    """
    Compute aggregated statistics (Solvability, Top-K) with bootstrap confidence intervals.

    Args:
        eval_results: The output from score_predictions().
        n_boot: Number of bootstrap iterations (default: 10,000).
        seed: Random seed for reproducibility.

    Returns:
        ModelStatistics object containing stratified metrics and CIs.
    """
    return analyze_workflow.compute_model_statistics(eval_results, n_boot=n_boot, seed=seed)
