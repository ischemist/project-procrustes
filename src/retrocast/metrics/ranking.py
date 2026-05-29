from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from retrocast.metrics.bootstrap import get_bootstrap_distribution

T = TypeVar("T")


@dataclass(frozen=True)
class RankResult:
    model_name: str
    rank_probs: dict[int, float]
    expected_rank: float


@dataclass(frozen=True)
class PairwiseComparison:
    metric: str
    model_a: str
    model_b: str
    diff_mean: float
    diff_ci_low: float
    diff_ci_high: float
    is_significant: bool
    count: int


def compute_probabilistic_ranking(
    model_results: dict[str, Sequence[T]],
    metric_extractor: Callable[[T], float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> list[RankResult]:
    model_names = sorted(model_results)
    n_models = len(model_names)
    if n_models == 0:
        return []

    boot_matrix = np.zeros((n_boot, n_models))
    for index, name in enumerate(model_names):
        boot_matrix[:, index] = get_bootstrap_distribution(
            model_results[name],
            metric_extractor,
            n_boot=n_boot,
            seed=seed + index,
        )

    ranks = np.argsort(np.argsort(-boot_matrix, axis=1), axis=1) + 1
    results = []
    for index, name in enumerate(model_names):
        model_ranks = ranks[:, index]
        probabilities = {rank: float(np.mean(model_ranks == rank)) for rank in range(1, n_models + 1)}
        expected_rank = sum(rank * probability for rank, probability in probabilities.items())
        results.append(RankResult(model_name=name, rank_probs=probabilities, expected_rank=expected_rank))
    return sorted(results, key=lambda result: (result.expected_rank, result.model_name))


def compute_pairwise_tournament(
    model_results: dict[str, Sequence[T]],
    metric_extractor: Callable[[T], float],
    metric_name: str,
    *,
    id_extractor: Callable[[T], str] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> list[PairwiseComparison]:
    comparisons = []
    models = sorted(model_results)
    for left_index, model_a in enumerate(models):
        for right_index, model_b in enumerate(models):
            if model_a == model_b:
                continue
            comparisons.append(
                compute_paired_difference(
                    model_results[model_a],
                    model_results[model_b],
                    metric_extractor,
                    model_a_name=model_a,
                    model_b_name=model_b,
                    metric_name=metric_name,
                    id_extractor=id_extractor,
                    n_boot=n_boot,
                    seed=seed + left_index * len(models) + right_index,
                )
            )
    return comparisons


def compute_paired_difference(
    targets_a: Sequence[T],
    targets_b: Sequence[T],
    metric_extractor: Callable[[T], float],
    *,
    model_a_name: str,
    model_b_name: str,
    metric_name: str,
    id_extractor: Callable[[T], str] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> PairwiseComparison:
    get_id = id_extractor or _default_target_id
    left = {get_id(target): target for target in targets_a}
    right = {get_id(target): target for target in targets_b}
    target_ids = sorted(set(left) & set(right))
    if not target_ids:
        raise ValueError("no common targets found between models")

    values_a = np.array([metric_extractor(left[target_id]) for target_id in target_ids])
    values_b = np.array([metric_extractor(right[target_id]) for target_id in target_ids])
    diffs = _paired_bootstrap_difference(values_a, values_b, n_boot=n_boot, seed=seed)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    diff_mean = float(np.mean(values_b) - np.mean(values_a))
    return PairwiseComparison(
        metric=metric_name,
        model_a=model_a_name,
        model_b=model_b_name,
        diff_mean=diff_mean,
        diff_ci_low=ci_low,
        diff_ci_high=ci_high,
        is_significant=not (ci_low <= 0 <= ci_high),
        count=len(target_ids),
    )


def _paired_bootstrap_difference(values_a: np.ndarray, values_b: np.ndarray, *, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values_a), (n_boot, len(values_a)))
    return np.mean(values_b[indices], axis=1) - np.mean(values_a[indices], axis=1)


def _default_target_id(target: object) -> str:
    target_id = getattr(target, "target_id", None)
    if isinstance(target_id, str):
        return target_id

    nested_target = getattr(target, "target", None)
    nested_id = getattr(nested_target, "id", None)
    if isinstance(nested_id, str):
        return nested_id

    raise TypeError("target id is not discoverable; pass id_extractor")
