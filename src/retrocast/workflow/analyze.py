import logging

from retrocast.metrics.bootstrap import (
    compute_metric_with_ci,
    get_has_stock_terminated_route,
    get_reciprocal_solv_rank,
    get_reciprocal_tier_rank,
    get_solv_i,
    make_get_top_k,
)
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
    has_lengths = any(t.stratification_length is not None for t in targets)
    group_fn = None
    if has_lengths:

        def _get_stratification_length(t: TargetEvaluation) -> str | None:
            if t.stratification_length is None:
                return None
            return f"depth {t.stratification_length}"

        group_fn = _get_stratification_length

    # --- 2. Stock termination / Solv-0 ---
    stat_stock_termination = compute_metric_with_ci(
        targets,
        get_has_stock_terminated_route,
        "Stock-Termination Rate",
        group_by=group_fn,
        n_boot=n_boot,
        seed=seed,
    )
    stat_tier_0_validity = None
    stat_solv_0 = None
    stat_mrr_tier_0 = None
    stat_mrr_solv_0 = None
    if any("tier 0" in t.first_valid_ranks for t in targets):
        stat_tier_0_validity = compute_metric_with_ci(
            targets,
            lambda target: 1.0 if target.first_valid_rank(tier=0) is not None else 0.0,
            "Tier-0 Validity",
            group_by=group_fn,
            n_boot=n_boot,
            seed=seed,
        )
        stat_mrr_tier_0 = compute_metric_with_ci(
            targets,
            get_reciprocal_tier_rank(0),
            "MRR Tier-0",
            group_by=group_fn,
            n_boot=n_boot,
            seed=seed,
        )
    if any("tier 0" in t.first_solv_ranks.get("stock", {}) for t in targets):
        stat_solv_0 = compute_metric_with_ci(
            targets,
            get_solv_i("stock", 0),
            "Solv-0[STR]",
            group_by=group_fn,
            n_boot=n_boot,
            seed=seed,
        )
        stat_mrr_solv_0 = compute_metric_with_ci(
            targets,
            get_reciprocal_solv_rank("stock", 0),
            "MRR Solv-0[STR]",
            group_by=group_fn,
            n_boot=n_boot,
            seed=seed,
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

    # --- 4. Aggregate Runtime Statistics ---
    total_wall_time = None
    total_cpu_time = None
    mean_wall_time = None
    mean_cpu_time = None

    wall_times = [t.wall_time for t in targets if t.wall_time is not None]
    cpu_times = [t.cpu_time for t in targets if t.cpu_time is not None]

    if wall_times:
        total_wall_time = sum(wall_times)
        mean_wall_time = total_wall_time / len(wall_times)
        logger.info(f"Runtime: {total_wall_time:.2f}s total, {mean_wall_time:.2f}s mean (wall time)")

    if cpu_times:
        total_cpu_time = sum(cpu_times)
        mean_cpu_time = total_cpu_time / len(cpu_times)
        logger.info(f"Runtime: {total_cpu_time:.2f}s total, {mean_cpu_time:.2f}s mean (CPU time)")

    return ModelStatistics(
        model_name=eval_results.model_name,
        benchmark=eval_results.benchmark_name,
        stock=eval_results.stock_name,
        stock_termination=stat_stock_termination,
        tier_0_validity=stat_tier_0_validity,
        solv_0=stat_solv_0,
        mrr_tier_0=stat_mrr_tier_0,
        mrr_solv_0=stat_mrr_solv_0,
        top_k_accuracy=stat_topk,
        total_wall_time=total_wall_time,
        total_cpu_time=total_cpu_time,
        mean_wall_time=mean_wall_time,
        mean_cpu_time=mean_cpu_time,
    )
