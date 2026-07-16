from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrocast.metrics.bootstrap import StratifiedMetricSummary
from retrocast.metrics.ranking import PairwiseComparison, RankResult
from retrocast.models.analysis import AnalysisReport, MetricSummary, ReliabilityFlag
from retrocast.visualization import adapters, model_performance, plots, theme
from retrocast.visualization.routes import RouteStats, create_route_comparison_figure


def metric(value: float, count: int = 40, ci_low: float | None = None, ci_high: float | None = None) -> MetricSummary:
    return MetricSummary(
        value=value,
        count=count,
        ci_low=ci_low,
        ci_high=ci_high,
        reliability=ReliabilityFlag(code="OK", message="Reliable."),
    )


def stratified_metric(overall: float, by_depth: dict[str, float]) -> StratifiedMetricSummary:
    return StratifiedMetricSummary(
        metric_name="test",
        overall=metric(overall, ci_low=overall - 0.1, ci_high=overall + 0.1),
        by_stratum={
            depth: metric(value, ci_low=value - 0.05, ci_high=value + 0.05) for depth, value in by_depth.items()
        },
    )


def stats(model_name: str, *, wall_time: float | None = 3600.0) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        solv_0=stratified_metric(0.6, {"depth 1": 0.5, "depth 2": 0.75}),
        top_k_accuracy={
            1: stratified_metric(0.4, {"depth 1": 0.25, "depth 2": 0.5}),
            5: stratified_metric(0.8, {"depth 1": 0.75, "depth 2": 1.0}),
            99: stratified_metric(0.9, {"depth 1": 0.9}),
        },
        total_wall_time=wall_time,
    )


@pytest.mark.unit
def test_visualization_adapters_preserve_metric_values_and_error_bars() -> None:
    series = adapters.stats_to_diagnostic_series(stats("model-a"))

    assert [item.name for item in series] == ["Solvability", "Top-1", "Top-5"]
    assert series[0].x == [1, 2]
    assert series[0].y == [50.0, 75.0]
    assert series[0].y_err_upper == pytest.approx([5.0, 5.0])
    assert series[0].custom_data == [[40, 45.0, 55.00000000000001, "OK"], [40, 70.0, 80.0, "OK"]]


@pytest.mark.unit
def test_visualization_adapters_support_overall_heatmap_and_stability_shapes() -> None:
    model_stats = [stats("model-a"), stats("model-b")]

    overall = adapters.stats_to_overall_series(model_stats, top_k_values=[1, 5, 10])
    heatmap = adapters.stats_to_heatmap_matrix(model_stats)
    stability = adapters.stats_to_stability_data(
        {"10": {"Top-1": metric(0.4, ci_low=0.3, ci_high=0.5)}, "2": {"Top-1": metric(0.6)}},
        "Top-1",
        "red",
    )

    assert overall[0].x == [0, 1, 2]
    assert overall[0].y == [60.0, 40.0, 80.0]
    assert heatmap.solvability.z == [[60.0], [60.0]]
    assert heatmap.top_k.x_labels == ["Top-1", "Top-5", "Top-99"]
    assert stability.seeds == ["2", "10"]
    assert stability.values == [60.0, 40.0]


@pytest.mark.unit
def test_metric_colors_are_stable_for_known_metric_names() -> None:
    assert theme.get_metric_color(" solvability ") == theme.COLOR_SOLVABILITY
    assert theme.get_metric_color("Top-1") == theme.COLOR_TOP_1
    assert theme.get_metric_color("top-5") == theme.COLOR_TOP_5
    assert theme.get_metric_color("top-10") == theme.COLOR_TOP_10
    assert theme.get_metric_color("top-not-a-number") == theme.COLOR_DEFAULT
    assert theme.get_model_color("same-model") == theme.get_model_color("same-model")


@pytest.mark.unit
def test_plot_functions_render_expected_trace_contracts() -> None:
    model_stats = [stats("model-a"), stats("model-b")]

    report_fig = plots.plot_analysis_report(AnalysisReport(metrics={"mrr": metric(0.25)}))
    diagnostic_fig = plots.plot_diagnostics(model_stats[0])
    comparison_fig = plots.plot_comparison(model_stats, metric_type="Top-K", k=1)
    overall_fig = plots.plot_overall_summary(model_stats, top_k_values=[1, 5])
    matrix_fig = plots.plot_performance_matrix(model_stats)

    assert report_fig.data[0].x == ("mrr",)
    assert [trace.name for trace in diagnostic_fig.data] == ["Solvability", "Top-1", "Top-5"]
    assert [trace.name for trace in comparison_fig.data] == ["model-a", "model-b"]
    assert overall_fig.layout.xaxis.ticktext == ("Solvability", "Top-1", "Top-5")
    assert len(matrix_fig.data) == 2
    assert matrix_fig.data[1].x == ("Top-1", "Top-5", "Top-99")


@pytest.mark.unit
def test_ranking_pairwise_stability_and_pareto_plots_encode_user_visible_values() -> None:
    ranking_fig = plots.plot_ranking(
        [RankResult(model_name="a", rank_probs={1: 0.75, 2: 0.25}, expected_rank=1.25)],
        "Top-1",
    )
    pairwise_fig = plots.plot_pairwise_matrix(
        [PairwiseComparison("Top-1", "a", "b", 0.1, 0.01, 0.2, True, 5)],
        "Top-1",
    )
    stability_fig = plots.plot_stability_analysis(
        [adapters.StabilityData("Top-1", ["1", "2"], [40.0, 60.0], [5.0, 0.0], [5.0, 0.0], 50.0, 10.0, "red")],
        "bench",
        "model-a",
    )
    pareto_fig = plots.plot_pareto_frontier(
        [stats("cheap"), stats("missing-cost"), stats("slow", wall_time=7200.0)],
        {"cheap": {"legend": "Cheap", "short": "C", "color": "red"}},
        {"cheap": 1.0, "slow": 2.0},
        k=1,
    )

    assert ranking_fig.data[0].z == ([0.75],)
    assert pairwise_fig.data[0].text[0][1] == "-10.0%*"
    assert stability_fig.data[0].x == (40.0, 60.0)
    assert [trace.name for trace in pareto_fig.data] == ["Cheap", "slow"]
    assert pareto_fig.data[0].x == (1.0,)


@pytest.mark.unit
def test_model_performance_wrappers_choose_public_plot_contracts() -> None:
    model_stats = [stats("model-a")]

    assert model_performance.plot_single_model_diagnostics(model_stats[0]).data
    assert (
        model_performance.plot_multi_model_comparison(model_stats, metric_type="Solvability").data[0].name == "model-a"
    )
    assert model_performance.plot_multi_model_comparison(model_stats, metric_type="Top-5").data[0].name == "model-a"
    assert model_performance.plot_overall_comparison(model_stats).data[0].name == "model-a"
    assert model_performance.plot_probabilistic_ranking([], "Top-1").data[0].z == ()


@pytest.mark.unit
def test_route_comparison_figure_counts_convergent_and_distribution_values() -> None:
    n1 = [
        RouteStats(depth=2, target_hac=10, target_mw=100.0, target_chiral=1, is_convergent=True),
        RouteStats(depth=3, target_hac=20, target_mw=200.0, target_chiral=0, is_convergent=False),
    ]
    n5 = [RouteStats(depth=2, target_hac=15, target_mw=150.0, target_chiral=2, is_convergent=True)]

    fig = create_route_comparison_figure(n1, n5)

    assert len(fig.data) == 10
    assert fig.data[0].y == (1, 1)
    assert fig.data[2].y == (1, 0)
    assert fig.data[-1].name == "n5 set"
