import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

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
    from retrocast import _native

    values = {name: [metric_extractor(target) for target in targets] for name, targets in model_results.items()}
    payload = json.loads(
        _native.probabilistic_ranking_json(
            json.dumps(values, separators=(",", ":")),
            n_boot,
            seed,
        )
    )
    return [
        RankResult(
            model_name=result["model_name"],
            rank_probs={int(rank): probability for rank, probability in result["rank_probs"].items()},
            expected_rank=result["expected_rank"],
        )
        for result in payload
    ]


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
    from retrocast import _native

    get_id = id_extractor or _default_target_id
    left = {get_id(target): target for target in targets_a}
    right = {get_id(target): target for target in targets_b}
    target_ids = sorted(set(left) & set(right))
    if not target_ids:
        raise ValueError("no common targets found between models")
    values_a = [metric_extractor(left[target_id]) for target_id in target_ids]
    values_b = [metric_extractor(right[target_id]) for target_id in target_ids]
    payload = json.loads(
        _native.paired_difference_json(
            json.dumps(values_a, separators=(",", ":")),
            json.dumps(values_b, separators=(",", ":")),
            model_a_name,
            model_b_name,
            metric_name,
            n_boot,
            seed,
        )
    )
    return PairwiseComparison(**payload)


def _default_target_id(target: object) -> str:
    target_id = getattr(target, "target_id", None)
    if isinstance(target_id, str):
        return target_id
    nested_target = getattr(target, "target", None)
    nested_id = getattr(nested_target, "id", None)
    if isinstance(nested_id, str):
        return nested_id
    raise TypeError("target id is not discoverable; pass id_extractor")
