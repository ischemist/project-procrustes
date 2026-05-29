from rich.console import Console

from retrocast.metrics.ranking import PairwiseComparison, RankResult
from retrocast.models.analysis import AnalysisReport, MetricSummary
from retrocast.visualization import (
    adapters,
    create_analysis_table,
    generate_markdown_report,
    model_performance,
    plots,
    theme,
)
from retrocast.visualization.report import create_ranking_table, create_stability_table, create_tournament_table


def test_visualization_report_uses_analysis_report() -> None:
    report = AnalysisReport(
        metrics={
            "solv_0[buyables]_rate": MetricSummary(value=0.5, count=2, ci_low=0.0, ci_high=1.0),
        }
    )

    markdown = generate_markdown_report(report)
    table = create_analysis_table(report)

    assert "Solv-0[buyables]" in markdown
    assert table.row_count > 0


def test_restored_visualization_modules_are_importable() -> None:
    assert theme.COLOR_TOP_1
    assert adapters.PlotSeries
    assert "plot_comparison" in plots.__all__
    assert model_performance.plot_single_model_diagnostics


def test_restored_ranking_and_tournament_tables() -> None:
    ranking_table = create_ranking_table(
        [RankResult(model_name="a", rank_probs={1: 0.75, 2: 0.25}, expected_rank=1.25)],
        "Top-1",
    )
    tournament_table = create_tournament_table(
        [
            PairwiseComparison(
                metric="Top-1",
                model_a="a",
                model_b="b",
                diff_mean=0.1,
                diff_ci_low=0.01,
                diff_ci_high=0.2,
                is_significant=True,
                count=3,
            )
        ],
        ["a", "b"],
    )
    stats_table, seed_table = create_stability_table(
        {"Top-1": {"mean": 50.0, "std": 1.0}},
        [("42", 0.0, 0.0, 0.0, 0.0)],
    )

    assert ranking_table.row_count == 1
    assert tournament_table.row_count == 2
    assert stats_table.row_count == 1
    assert seed_table.row_count == 1


def test_tournament_table_displays_row_minus_column() -> None:
    table = create_tournament_table(
        [
            PairwiseComparison(
                metric="Top-1",
                model_a="row",
                model_b="column",
                diff_mean=0.1,
                diff_ci_low=0.01,
                diff_ci_high=0.2,
                is_significant=True,
                count=3,
            )
        ],
        ["row", "column"],
    )
    console = Console(record=True, width=80)
    console.print(table)

    rendered = console.export_text()
    assert "-10.0%" in rendered
