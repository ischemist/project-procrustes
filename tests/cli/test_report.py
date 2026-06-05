from __future__ import annotations

from rich.console import Console

from retrocast.cli.report import create_analysis_table, generate_markdown_report
from retrocast.models.analysis import AnalysisReport, MetricSummary, ReliabilityFlag, RuntimeSummary


def report_with_all_metric_groups() -> AnalysisReport:
    return AnalysisReport(
        metrics={
            "tier_0_validity_rate": MetricSummary(value=0.75, count=4, ci_low=0.5, ci_high=1.0),
            "solv_0[test-stock]_rate": MetricSummary(value=1.0, count=4, ci_low=0.8, ci_high=1.0),
            "tier_0_validity_mrr": MetricSummary(value=0.25, count=4, ci_low=0.0, ci_high=0.5),
            "solv_0[test-stock]_mrr": MetricSummary(value=0.5, count=4, ci_low=0.25, ci_high=0.75),
            "acceptable_reconstruction_top_3[test-stock]": MetricSummary(
                value=0.75,
                count=4,
                reliability=ReliabilityFlag(code="LOW_N", message="Small sample size."),
            ),
            "distinct_root_reactions_top_3[test-stock]": MetricSummary(value=2.0, count=4),
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
    assert "Tier-0 Validity" in markdown
    assert "Solv-0[test-stock]" in markdown
    assert "MRR Tier-0" in markdown
    assert "MRR Solv-0[test-stock]" in markdown
    assert "| MRR Tier-0 | 0.250 | [0.000, 0.500] | 4 |  |" in markdown
    assert "| MRR Solv-0[test-stock] | 0.500 | [0.250, 0.750] | 4 |  |" in markdown
    assert "Top-3" in markdown
    assert "Mean distinct roots" in markdown
    assert "| Top-3 | 75.0%! |  |  | 2.000 |" in markdown
    assert "## By Stratum" in markdown
    assert "### depth=2" in markdown
    assert "| Top-1 | 25.0% |  | 2 |  |" in markdown
    assert "| Top-3 | 75.0% |  | 4 | LOW_N |" in markdown


def test_create_analysis_table_renders_metric_groups() -> None:
    console = Console(record=True, width=120)
    console.print(create_analysis_table(report_with_all_metric_groups()))

    output = console.export_text()
    assert "Solv-N Evaluation" in output
    assert "test-stock" in output
    assert "Benchmark Route" in output
    assert "Top-3" in output
    assert "75.0%" in output
    assert "0.250" in output
    assert "0.500" in output
    assert "[0.000, 0.500]" in output
    assert "[0.250, 0.750]" in output
    assert "Mean distinct roots" in output
    assert "2.000" in output
    assert "flags: ! low n / unstable ci" in output


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
    assert "| Total time | 12.00 s | 4.00 s |" in markdown
    assert "| Per target | 3.00 s | 1.00 s |" in markdown

    console = Console(record=True, width=120)
    console.print(create_analysis_table(report))
    output = console.export_text()
    assert "Runtime" in output
    assert "Total time" in output
    assert "12.00 s" in output
