from __future__ import annotations

import pytest

from retrocast.models.stats import MetricResult, ReliabilityFlag, StratifiedMetric
from retrocast.visualization.depth import depth_group_sort_key, depth_group_value
from retrocast.visualization.report import format_metric_table


def _result(value: float) -> MetricResult:
    return MetricResult(
        value=value,
        ci_lower=value,
        ci_upper=value,
        n_samples=1,
        reliability=ReliabilityFlag(code="OK", message="ok"),
    )


@pytest.mark.unit
def test_depth_group_keys_sort_numerically() -> None:
    keys = ["depth 10", "2", "depth 1", 3]

    assert sorted(keys, key=depth_group_sort_key) == ["depth 1", "2", 3, "depth 10"]
    assert [depth_group_value(key) for key in ["depth 1", "2", 3]] == [1, 2, 3]


@pytest.mark.unit
def test_metric_table_sorts_depth_labels_numerically() -> None:
    metric = StratifiedMetric(
        metric_name="solv_0",
        overall=_result(1.0),
        by_group={
            "depth 10": _result(0.1),
            "depth 2": _result(0.2),
        },
    )

    rendered = format_metric_table(metric)

    assert rendered.index("| depth 2 ") < rendered.index("| depth 10 ")
