from __future__ import annotations

from rich.console import Console

from retrocast.cli.report import create_analysis_table, generate_markdown_report
from retrocast.models.analysis import AnalysisReport, MetricSummary, RuntimeSummary


def report_with_all_metric_groups() -> AnalysisReport:
    return AnalysisReport(
        metrics={
            "solv_0[test-stock]_rate": MetricSummary(value=1.0, count=4, ci_low=0.8, ci_high=1.0),
            "mrr_solv_0[test-stock]": MetricSummary(value=0.5, count=4, ci_low=0.25, ci_high=0.75),
            "acceptable_reconstruction_top_3[test-stock]": MetricSummary(value=0.75, count=4),
        },
        by_stratum={
            "depth=2": {
                "acceptable_reconstruction_top_1[test-stock]": MetricSummary(value=0.25, count=2),
            }
        },
    )


def test_generate_markdown_report_includes_strata_and_metric_groups() -> None:
    markdown = generate_markdown_report(report_with_all_metric_groups(), title="Small Run")

    assert "# Small Run" in markdown
    assert "Solv-0[test-stock]" in markdown
    assert "MRR Solv-0[test-stock]" in markdown
    assert "Top-3" in markdown
    assert "## By Stratum" in markdown
    assert "### depth=2" in markdown
    assert "| Top-1 | 25.0% |  | 2 |" in markdown


def test_create_analysis_table_renders_top_k_without_ci() -> None:
    console = Console(record=True, width=120)
    console.print(create_analysis_table(report_with_all_metric_groups()))

    output = console.export_text()
    assert "Benchmark route reconstruction" in output
    assert "Top-3" in output
    assert "75.0%" in output


def test_reports_render_runtime_summary() -> None:
    report = report_with_all_metric_groups().model_copy(
        update={
            "runtime": RuntimeSummary(
                total_wall_time=12.0,
                mean_wall_time=3.0,
                total_cpu_time=4.0,
                mean_cpu_time=1.0,
                timed_target_count=4,
            )
        }
    )

    markdown = generate_markdown_report(report, title="Small Run")
    assert "## Runtime" in markdown
    assert "| Total wall time | 12.00s | 4 |" in markdown

    console = Console(record=True, width=120)
    console.print(create_analysis_table(report))
    output = console.export_text()
    assert "Runtime" in output
    assert "Total wall time" in output
    assert "12.00s" in output
