from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retrocast.models.analysis import AnalysisReport
from retrocast.visualization import adapters, theme
from retrocast.visualization.adapters import PlotSeries


def plot_analysis_report(report: AnalysisReport, *, title: str = "Analysis Report"):
    names = list(report.metrics)
    values = [report.metrics[name].value for name in names]
    fig = go.Figure(data=[go.Bar(x=names, y=values)])
    fig.update_layout(title=title, yaxis_title="value")
    return fig


def plot_diagnostics(stats: Any):
    fig = go.Figure()
    for series in adapters.stats_to_diagnostic_series(stats):
        _render_series(fig, series)
    title = f"Performance Diagnostics: {stats.model_name}"
    theme.apply_layout(fig, title=title, x_title="Route Depth", y_title="Percentage (%)")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_comparison(models_stats: list[Any], metric_type: str = "Top-K", k: int = 1):
    series_list = adapters.stats_to_comparison_series(models_stats, metric_type, k)
    offsets = _calculate_offsets(len(series_list), width=0.6)
    fig = go.Figure()
    all_x = set()
    for index, series in enumerate(series_list):
        all_x.update(series.x)
        _render_series(fig, series, x_offset=offsets[index])
    title_suffix = f" (k={k})" if metric_type == "Top-K" else ""
    theme.apply_layout(
        fig, title=f"Model Comparison: {metric_type}{title_suffix}", x_title="Route Depth", y_title="Percentage (%)"
    )
    if all_x:
        depths = sorted(all_x)
        fig.update_xaxes(tickmode="array", tickvals=depths, ticktext=[f"Depth {int(depth)}" for depth in depths])
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_overall_summary(models_stats: list[Any], top_k_values: list[int] | None = None):
    top_k_values = top_k_values or [1, 2, 3, 4, 5, 10, 20, 50]
    series_list = adapters.stats_to_overall_series(models_stats, top_k_values)
    offsets = _calculate_offsets(len(series_list), width=0.6)
    fig = go.Figure()
    for index, series in enumerate(series_list):
        _render_series(fig, series, x_offset=offsets[index])
    labels = ["Solvability"] + [f"Top-{k}" for k in top_k_values]
    theme.apply_layout(fig, title="Overall Performance Summary", x_title="Metric", y_title="Percentage (%)")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels)
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_performance_matrix(models_stats: list[Any]):
    data = adapters.stats_to_heatmap_matrix(models_stats)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.12, 0.88],
        horizontal_spacing=0.03,
        subplot_titles=(data.solvability.title, data.top_k.title),
    )
    fig.add_trace(
        go.Heatmap(
            z=data.solvability.z,
            x=data.solvability.x_labels,
            y=data.solvability.y_labels,
            text=data.solvability.text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            coloraxis="coloraxis1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=data.top_k.z,
            x=data.top_k.x_labels,
            y=data.top_k.y_labels,
            text=data.top_k.text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            coloraxis="coloraxis2",
        ),
        row=1,
        col=2,
    )
    max_top = max((value for row in data.top_k.z for value in row if value is not None), default=0) or 100
    theme.apply_layout(fig, legend_top=False, height=400 + len(data.solvability.y_labels) * 20, width=1000)
    fig.update_layout(
        coloraxis1={"colorscale": "Greens", "showscale": False, "cmin": 0, "cmax": 100},
        coloraxis2={"colorscale": "Blues", "showscale": False, "cmin": 0, "cmax": max_top},
    )
    fig.update_yaxes(title_text="Model", row=1, col=1)
    return fig


def plot_ranking(ranking: list[Any], metric_name: str):
    y_labels = [result.model_name for result in ranking][::-1]
    n_models = len(ranking)
    x_labels = [f"Rank {index}" for index in range(1, n_models + 1)]
    z_values = []
    text_values = []
    for result in ranking[::-1]:
        row = [result.rank_probs.get(rank, 0.0) for rank in range(1, n_models + 1)]
        z_values.append(row)
        text_values.append([f"{value:.0%}" if value > 0.01 else "" for value in row])
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            xgap=1,
            ygap=1,
        )
    )
    theme.apply_layout(
        fig, title=f"Probabilistic Ranking: {metric_name}", x_title="Rank", y_title="Model", legend_top=False
    )
    return fig


def plot_pairwise_matrix(comparisons: list[Any], metric_name: str):
    models = sorted(
        {comparison.model_a for comparison in comparisons} | {comparison.model_b for comparison in comparisons}
    )
    model_map = {model: index for index, model in enumerate(models)}
    n_models = len(models)
    z_values = [[None] * n_models for _ in range(n_models)]
    text_values = [["" for _ in range(n_models)] for _ in range(n_models)]
    custom_data = [[None for _ in range(n_models)] for _ in range(n_models)]
    max_diff = 0.0
    for comparison in comparisons:
        row = model_map[comparison.model_a]
        col = model_map[comparison.model_b]
        diff = -comparison.diff_mean
        max_diff = max(max_diff, abs(diff))
        z_values[row][col] = diff
        text_values[row][col] = f"{diff:+.1%}{'*' if comparison.is_significant else ''}"
        custom_data[row][col] = [comparison.model_a, comparison.model_b, comparison.count]
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=models,
            y=models,
            text=text_values,
            texttemplate="%{text}",
            customdata=custom_data,
            colorscale="RdBu",
            zmid=0,
            zmin=-max_diff,
            zmax=max_diff,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{customdata[0]} vs %{customdata[1]}</b><br>diff: %{z:+.2%}<br>N=%{customdata[2]}<extra></extra>",
        )
    )
    theme.apply_layout(
        fig,
        title=f"Pairwise Comparison Matrix: {metric_name}",
        x_title="Opponent",
        y_title="Model",
        height=600 + n_models * 30,
        width=700 + n_models * 30,
        legend_top=False,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_stability_analysis(data_list: list[adapters.StabilityData], bench_name: str, model_name: str):
    fig = go.Figure()
    for data in data_list:
        fig.add_trace(
            go.Scatter(
                x=data.values,
                y=[f"Seed {seed}" for seed in data.seeds],
                name=data.metric_name,
                mode="markers",
                marker={"color": data.color, "size": 8},
                error_x={"type": "data", "array": data.errors_plus, "arrayminus": data.errors_minus, "visible": True},
                hovertemplate=f"<b>{data.metric_name}</b><br>%{{y}}<br>Value: %{{x:.2f}}%<extra></extra>",
            )
        )
        fig.add_vline(
            x=data.grand_mean,
            line_width=2,
            line_dash="dot",
            line_color=data.color,
            annotation_text=f"mean={data.grand_mean:.1f}%",
            annotation_position="top right",
        )
    theme.apply_layout(
        fig,
        title=f"[{bench_name}] Stability Analysis ({model_name})",
        x_title="Performance (%)",
        y_title="Seed Variant",
        height=max(600, len(data_list[0].seeds) * 25) if data_list else 600,
    )
    fig.update_xaxes(dtick=10, range=[0, 100])
    return fig


def plot_pareto_frontier(
    models_stats: list[Any],
    model_config: dict[str, dict[str, str]],
    hourly_costs: dict[str, float],
    k: int = 10,
    time_based: bool = False,
):
    fig = go.Figure()
    pareto_points: list[tuple[float, float]] = []
    for stats in models_stats:
        model_name = stats.model_name
        wall_time = getattr(stats, "total_wall_time", None)
        if wall_time is None or k not in getattr(stats, "top_k_accuracy", {}):
            continue
        if not time_based and model_name not in hourly_costs:
            continue
        metric = stats.top_k_accuracy[k].overall
        x_value = wall_time / 60 if time_based else wall_time / 3600 * hourly_costs[model_name]
        accuracy = metric.value * 100
        ci_low = (metric.ci_low if metric.ci_low is not None else metric.value) * 100
        ci_high = (metric.ci_high if metric.ci_high is not None else metric.value) * 100
        config = model_config.get(
            model_name, {"legend": model_name, "short": model_name[:10], "color": theme.get_model_color(model_name)}
        )
        pareto_points.append((x_value, accuracy))
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[accuracy],
                name=config["legend"],
                mode="markers+text",
                marker={"color": config["color"], "size": 12, "line": {"width": 1, "color": "white"}},
                text=[config["short"]],
                textposition="middle right",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [ci_high - accuracy],
                    "arrayminus": [accuracy - ci_low],
                    "visible": True,
                },
                customdata=[[model_name, metric.count, ci_low, ci_high]],
                hovertemplate="<b>%{customdata[0]}</b><br>x=%{x:.2f}<br>Top-K=%{y:.1f}%<br>CI=[%{customdata[2]:.1f}%, %{customdata[3]:.1f}%]<br>N=%{customdata[1]}<extra></extra>",
            )
        )
    pareto_points.sort(key=lambda point: point[0])
    frontier = []
    best_accuracy = -float("inf")
    for x_value, accuracy in pareto_points:
        if accuracy > best_accuracy:
            frontier.append((x_value, accuracy))
            best_accuracy = accuracy
    if len(frontier) > 1:
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in frontier],
                y=[point[1] for point in frontier],
                mode="lines",
                name="Pareto Frontier",
                line={"color": "rgba(128,128,128,0.5)", "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    x_title = "Wall Time (minutes)" if time_based else "Total Cost (USD)"
    theme.apply_layout(fig, x_title=x_title, y_title=f"Top-{k} Accuracy (%)", height=600, width=1200)
    fig.update_yaxes(range=[0, 100])
    return fig


def _render_series(fig: Any, series: PlotSeries, x_offset: float = 0.0) -> None:
    x_values = [value + x_offset if isinstance(value, int | float) else value for value in series.x]
    error_y = None
    if series.y_err_upper:
        error_y = {
            "type": "data",
            "symmetric": False,
            "array": series.y_err_upper,
            "arrayminus": series.y_err_lower,
            "visible": True,
        }
    common_args = {
        "name": series.name,
        "x": x_values,
        "y": series.y,
        "marker_color": series.color,
        "customdata": series.custom_data,
        "hovertemplate": "<b>%{fullData.name}</b><br>Value: %{y:.1f}%<br>N=%{customdata[0]}<br>CI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>Status: %{customdata[3]}<extra></extra>",
    }
    if series.mode_hint == "bar":
        fig.add_trace(go.Bar(**common_args, error_y=error_y))
        return
    if error_y is not None:
        error_y = {**error_y, "width": 4, "thickness": 1.5}
    fig.add_trace(
        go.Scatter(**common_args, mode="markers", error_y=error_y, marker={"color": series.color, "size": 10})
    )


def _calculate_offsets(n_items: int, width: float = 0.6) -> list[float]:
    if n_items <= 1:
        return [0.0]
    step = width / (n_items - 1)
    return [-width / 2 + index * step for index in range(n_items)]


__all__ = [
    "plot_analysis_report",
    "plot_comparison",
    "plot_diagnostics",
    "plot_overall_summary",
    "plot_pairwise_matrix",
    "plot_pareto_frontier",
    "plot_performance_matrix",
    "plot_ranking",
    "plot_stability_analysis",
]
