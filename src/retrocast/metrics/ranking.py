import logging
from collections.abc import Callable

import numpy as np

from retrocast.metrics.bootstrap import compute_paired_difference, get_bootstrap_distribution
from retrocast.models.evaluation import EvaluationResults, TargetEvaluation
from retrocast.models.stats import ModelComparison, RankResult

logger = logging.getLogger(__name__)


def compute_probabilistic_ranking(
    model_results: dict[str, EvaluationResults],
    metric_extractor: Callable[[TargetEvaluation], float],
    n_boot: int = 10000,
    seed: int = 42,
) -> list[RankResult]:
    # Ensure consistent order
    model_names = sorted(model_results.keys())
    n_models = len(model_names)

    # 1. Generate Bootstrap Matrix (n_boot, n_models)
    logger.info("Generating bootstrap distributions...")
    boot_matrix = np.zeros((n_boot, n_models))

    for i, name in enumerate(model_names):
        boot_matrix[:, i] = get_bootstrap_distribution(
            list(model_results[name].results.values()), metric_extractor, n_boot=n_boot, seed=seed
        )

    # 2. Calculate Ranks (Pure Numpy)
    logger.info("Simulating tournament...")

    # We want descending sort (higher score = rank 1).
    # np.argsort sorts ascending. So we sort -boot_matrix.
    # argsort gives indices. argsort of argsort gives ranks (0-based).
    # axis=1 sorts along the model dimension for each bootstrap sample.

    # Example: Scores [0.8, 0.9, 0.7] -> Neg [-0.8, -0.9, -0.7]
    # Argsort: [1, 0, 2] (indices of max to min)
    # Argsort of Argsort: [1, 0, 2] -> Ranks: Model 0 is rank 1, Model 1 is rank 0, Model 2 is rank 2.

    temp_sort = np.argsort(-boot_matrix, axis=1)
    ranks_0based = np.argsort(temp_sort, axis=1)
    ranks_1based = ranks_0based + 1

    # 3. Aggregate Probabilities
    results = []
    for i, name in enumerate(model_names):
        # Extract this model's column of ranks
        model_ranks = ranks_1based[:, i]

        # Count occurrences of each rank
        unique, counts = np.unique(model_ranks, return_counts=True)
        counts_dict = dict(zip(unique, counts, strict=True))

        rank_probs = {}
        expected_sum = 0.0

        for r in range(1, n_models + 1):
            count = counts_dict.get(r, 0)
            prob = count / n_boot
            rank_probs[r] = prob
            expected_sum += r * prob

        results.append(RankResult(model_name=name, rank_probs=rank_probs, expected_rank=expected_sum))

    results.sort(key=lambda x: x.expected_rank)
    return results


def compute_pairwise_tournament(
    model_results: dict[str, EvaluationResults],
    metric_extractor: Callable[[TargetEvaluation], float],
    metric_name: str,
    n_boot: int = 10000,
) -> list[ModelComparison]:
    """
    Runs a round-robin tournament where every model plays every other model.
    Returns a flat list of pairwise comparisons.
    """
    models = sorted(model_results.keys())
    comparisons = []

    logger.info(f"Starting pairwise tournament for {len(models)} models...")

    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                continue

            # We run A vs B.
            # (Optimization: We could just invert A vs B to get B vs A,
            # but running it explicitly is safer and cheap enough).

            # Get target lists (aligned by ID inside compute_paired_difference)
            targets_a = list(model_results[model_a].results.values())
            targets_b = list(model_results[model_b].results.values())

            try:
                comp = compute_paired_difference(
                    targets_a, targets_b, metric_extractor, model_a, model_b, metric_name, n_boot=n_boot
                )
                comparisons.append(comp)
            except ValueError:
                logger.warning(f"Could not compare {model_a} vs {model_b}")

    return comparisons
