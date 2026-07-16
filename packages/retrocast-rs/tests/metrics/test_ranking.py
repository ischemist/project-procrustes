from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrocast.metrics.ranking import (
    compute_paired_difference,
    compute_pairwise_tournament,
    compute_probabilistic_ranking,
)


@dataclass(frozen=True)
class Sample:
    target_id: str
    value: float


def test_probabilistic_ranking_orders_models_by_metric() -> None:
    results = {
        "weak": samples([0, 0, 1, 0]),
        "strong": samples([1, 1, 1, 1]),
        "middle": samples([1, 0, 1, 1]),
    }

    ranking = compute_probabilistic_ranking(results, lambda sample: sample.value, n_boot=200, seed=3)

    assert [result.model_name for result in ranking] == ["strong", "middle", "weak"]
    assert ranking[0].rank_probs[1] > 0.5
    assert all(abs(sum(result.rank_probs.values()) - 1.0) < 1e-9 for result in ranking)


def test_paired_difference_aligns_by_target_id() -> None:
    left = [Sample("b", 0.0), Sample("a", 0.0)]
    right = [Sample("a", 1.0), Sample("b", 1.0)]

    comparison = compute_paired_difference(
        left,
        right,
        lambda sample: sample.value,
        model_a_name="left",
        model_b_name="right",
        metric_name="toy",
        n_boot=100,
    )

    assert comparison.diff_mean == 1.0
    assert comparison.count == 2
    assert comparison.is_significant


def test_pairwise_tournament_returns_directed_comparisons() -> None:
    results = {
        "a": samples([1, 1, 1]),
        "b": samples([0, 0, 0]),
        "c": samples([1, 0, 0]),
    }

    comparisons = compute_pairwise_tournament(results, lambda sample: sample.value, "toy", n_boot=100)

    assert len(comparisons) == 6
    by_pair = {(comparison.model_a, comparison.model_b): comparison for comparison in comparisons}
    assert by_pair[("a", "b")].diff_mean == pytest.approx(-1.0)
    assert by_pair[("b", "a")].diff_mean == pytest.approx(1.0)


def test_paired_difference_requires_discoverable_target_ids() -> None:
    with pytest.raises(TypeError, match="id_extractor"):
        compute_paired_difference(
            [object()],
            [object()],
            lambda _: 0.0,
            model_a_name="a",
            model_b_name="b",
            metric_name="toy",
        )


def samples(values: list[float]) -> list[Sample]:
    return [Sample(target_id=f"target-{index}", value=value) for index, value in enumerate(values)]
