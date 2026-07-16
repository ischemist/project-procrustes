from __future__ import annotations

import numpy as np

from retrocast.metrics.bootstrap import (
    check_reliability,
    compute_metric_with_ci,
    get_bootstrap_distribution,
    summarize_values,
)


def test_reliability_flags_low_n() -> None:
    result = check_reliability(n=10, p=0.5)

    assert result.code == "LOW_N"


def test_reliability_flags_extreme_proportion() -> None:
    result = check_reliability(n=50, p=0.98)

    assert result.code == "EXTREME_P"


def test_reliability_flags_ok_for_adequate_sample() -> None:
    result = check_reliability(n=100, p=0.5)

    assert result.code == "OK"


def test_summarize_values_returns_ci_and_reliability() -> None:
    result = summarize_values([0.0, 1.0] * 20, n_boot=500, seed=4)

    assert result.value == 0.5
    assert result.count == 40
    assert result.ci_low is not None
    assert result.ci_high is not None
    assert result.ci_low <= result.value <= result.ci_high
    assert result.reliability is not None
    assert result.reliability.code == "OK"


def test_summarize_values_handles_empty_input() -> None:
    result = summarize_values([], n_boot=10)

    assert result.value == 0.0
    assert result.count == 0
    assert result.ci_low is None
    assert result.ci_high is None
    assert result.reliability is not None
    assert result.reliability.code == "LOW_N"


def test_compute_metric_with_ci_groups_values() -> None:
    result = compute_metric_with_ci(
        [1, 2, 3, 4],
        lambda value: float(value > 2),
        "toy",
        stratify_by=lambda value: "small" if value <= 2 else "large",
        n_boot=100,
    )

    assert result.metric_name == "toy"
    assert result.overall.value == 0.5
    assert result.by_stratum["small"].value == 0.0
    assert result.by_stratum["large"].value == 1.0


def test_get_bootstrap_distribution_shape_and_seed_stability() -> None:
    values = [1, 0, 1, 0]

    left = get_bootstrap_distribution(values, float, n_boot=100, seed=7)
    right = get_bootstrap_distribution(values, float, n_boot=100, seed=7)

    assert left.shape == (100,)
    assert np.array_equal(left, right)
