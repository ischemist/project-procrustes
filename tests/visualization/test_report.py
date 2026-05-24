from __future__ import annotations

import pytest
from rich.console import Console

from retrocast.models.stats import MetricResult, ModelStatistics, ReliabilityFlag, StratifiedMetric
from retrocast.visualization.report import create_single_model_summary_table, generate_markdown_report


def _metric(name: str, value: float) -> StratifiedMetric:
    return StratifiedMetric(
        metric_name=name,
        overall=MetricResult(
            value=value,
            ci_lower=value,
            ci_upper=value,
            n_samples=3,
            reliability=ReliabilityFlag(code="OK", message="ok"),
        ),
        by_group={},
    )


@pytest.mark.unit
def test_single_model_summary_table_includes_solv_n_metrics() -> None:
    stats = ModelStatistics(
        model_name="model",
        benchmark="benchmark",
        stock="stock",
        stock_termination=_metric("stock_termination", 0.5),
        tier_0_validity=_metric("tier_0_validity", 0.75),
        solv_0=_metric("solv_0", 0.25),
        mrr_tier_0=_metric("mrr_tier_0", 1 / 3),
        mrr_solv_0=_metric("mrr_solv_0", 1 / 6),
        top_k_accuracy={1: _metric("top_1", 1.0)},
    )

    table = create_single_model_summary_table(stats, visible_k=[1])
    console = Console(record=True, width=120)
    console.print(table)
    rendered = console.export_text()

    assert "Stock-Termination Rate" not in rendered
    assert "Solv-N hierarchy" in rendered
    assert "Tier-0 Validity" in rendered
    assert "Solv-0[STR]" in rendered
    assert "Rank within Solv-N hierarchy" in rendered
    assert "MRR Tier-0" in rendered
    assert "MRR Solv-0[STR]" in rendered
    assert "Benchmark route reconstruction" in rendered
    assert "Top-1" in rendered
    assert "0.333" in rendered


@pytest.mark.unit
def test_markdown_report_includes_solv_n_sections() -> None:
    stats = ModelStatistics(
        model_name="model",
        benchmark="benchmark",
        stock="stock",
        stock_termination=_metric("stock_termination", 0.5),
        tier_0_validity=_metric("tier_0_validity", 0.75),
        solv_0=_metric("solv_0", 0.25),
        mrr_tier_0=_metric("mrr_tier_0", 1 / 3),
        mrr_solv_0=_metric("mrr_solv_0", 1 / 6),
        top_k_accuracy={1: _metric("top_1", 1.0), 5: _metric("top_5", 1.0)},
    )

    report = generate_markdown_report(stats)

    assert "## Stock-Termination Rate" not in report
    assert "## Solv-N Hierarchy" in report
    assert "### Tier-0 Validity" in report
    assert "### Solv-0[STR]" in report
    assert "## Rank Within Solv-N Hierarchy" in report
    assert "### MRR Tier-0" in report
    assert "### MRR Solv-0[STR]" in report
    assert "## Benchmark Route Reconstruction" in report
    assert report.count("## Benchmark Route Reconstruction") == 1
    assert "### Top-1 Accuracy" in report
    assert "### Top-5 Accuracy" in report
    assert "**Overall**: 0.333" in report


@pytest.mark.unit
def test_runtime_durations_are_displayed_in_larger_units() -> None:
    stats = ModelStatistics(
        model_name="model",
        benchmark="benchmark",
        stock="stock",
        stock_termination=_metric("stock_termination", 0.5),
        top_k_accuracy={},
        total_wall_time=90,
        mean_wall_time=2.5,
        total_cpu_time=7200,
        mean_cpu_time=65,
    )

    table = create_single_model_summary_table(stats)
    console = Console(record=True, width=120)
    console.print(table)
    rendered = console.export_text()
    report = generate_markdown_report(stats)

    assert "1.5 min" in rendered
    assert "2.50 s" in rendered
    assert "2.0 h" in rendered
    assert "1.1 min" in rendered
    assert "**Total Wall Time**: 1.5 min" in report
    assert "**Mean Wall Time**: 2.50 s per target" in report
    assert "**Total CPU Time**: 2.0 h" in report
    assert "**Mean CPU Time**: 1.1 min per target" in report
