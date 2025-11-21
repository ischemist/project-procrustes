import logging

from retrocast.metrics.bootstrap import compute_metric_with_ci, get_is_solvable, make_get_top_k
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelStatistics

logger = logging.getLogger(__name__)


def compute_model_statistics(eval_results: EvaluationResults, n_boot: int = 10000, seed: int = 42) -> ModelStatistics:
    """
    Core workflow: Turns raw scored targets into bootstrapped statistics.
    """
    logger.info(f"Computing statistics for {eval_results.model_name}...")

    targets = list(eval_results.results.values())

    # Grouping key for stratification
    def get_length(t):
        return t.route_length

    # 1. Solvability
    stat_solvability = compute_metric_with_ci(
        targets, get_is_solvable, "Solvability", group_by=get_length, n_boot=n_boot, seed=seed
    )

    # 2. Top-K
    stat_topk = {}
    # calculating many K is cheap, we just filter what we display later
    for k in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        stat_topk[k] = compute_metric_with_ci(
            targets, make_get_top_k(k), f"Top-{k}", group_by=get_length, n_boot=n_boot, seed=seed
        )

    return ModelStatistics(
        model_name=eval_results.model_name,
        benchmark=eval_results.benchmark_name,
        stock=eval_results.stock_name,
        solvability=stat_solvability,
        top_k_accuracy=stat_topk,
    )
