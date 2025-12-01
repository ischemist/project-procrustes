import logging

from retrocast.metrics.bootstrap import compute_metric_with_ci, get_is_solvable, make_get_top_k
from retrocast.models.evaluation import EvaluationResults, TargetEvaluation
from retrocast.models.stats import ModelStatistics, StratifiedMetric

logger = logging.getLogger(__name__)


def compute_model_statistics(eval_results: EvaluationResults, n_boot: int = 10000, seed: int = 42) -> ModelStatistics:
    """
    Core workflow: Turns raw scored targets into bootstrapped statistics.

    Stratification is based on the properties of the MATCHED acceptable route,
    not pre-computed target metadata. This ensures metrics accurately reflect
    which routes the model actually found.
    """
    logger.info(f"Computing statistics for {eval_results.model_name}...")

    targets = list(eval_results.results.values())
    has_lengths = any(t.matched_route_length is not None for t in targets)
    group_fn = None
    if has_lengths:

        def _get_matched_length(t: TargetEvaluation) -> int:
            # Use -1 as a safe fallback for targets without matches
            # to ensure keys are always integers (required for visualization sorting).
            return t.matched_route_length if t.matched_route_length is not None else -1

        group_fn = _get_matched_length

    # --- 2. Solvability ---
    # This is always calculable as long as we have a stock.
    stat_solvability = compute_metric_with_ci(
        targets, get_is_solvable, "Solvability", group_by=group_fn, n_boot=n_boot, seed=seed
    )

    # --- 3. Top-K Accuracy ---
    # Only calculate this if the benchmark actually has acceptable routes.
    # If this is a pure prediction benchmark (no acceptable routes), we skip Top-K metrics.
    # Note: We check if the *benchmark* has acceptable routes, not if the *model* found any matches.
    stat_topk: dict[int, StratifiedMetric] = {}
    if eval_results.has_acceptable_routes:
        # calculating many K is cheap, we just filter what we display later
        for k in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000, 10000]:
            stat_topk[k] = compute_metric_with_ci(
                targets, make_get_top_k(k), f"Top-{k}", group_by=group_fn, n_boot=n_boot, seed=seed
            )
    else:
        logger.info("Benchmark has no acceptable routes. Skipping Top-K metrics.")

    return ModelStatistics(
        model_name=eval_results.model_name,
        benchmark=eval_results.benchmark_name,
        stock=eval_results.stock_name,
        solvability=stat_solvability,
        top_k_accuracy=stat_topk,
    )
