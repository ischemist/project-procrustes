"""
Plot generation functions.

This module handles the rendering of data into Plotly figures.
It relies on `retrocast.visualization.adapters` for data transformation
and `retrocast.visualization.theme` for styling.
"""

import plotly.graph_objects as go

from retrocast.models.stats import ModelComparison, ModelStatistics, RankResult
from retrocast.visualization import adapters, theme
from retrocast.visualization.adapters import PlotSeries

# --- Main Plotting Functions ---


def plot_diagnostics(stats: ModelStatistics) -> go.Figure:
    """
    Plots performance diagnostics for a single model (Solvability & Top-K vs Depth).
    Uses grouped bar charts.
    """
    # 1. Transform Data
    series_list = adapters.stats_to_diagnostic_series(stats)

    # 2. Initialize Figure
    fig = go.Figure()

    # 3. Render Traces
    for series in series_list:
        _render_series(fig, series)

    # 4. Apply Theme
    full_title = (
        f"<b>Performance Diagnostics: {stats.model_name}</b><br>"
        f"<span style='font-size: 12px;'>Benchmark: {stats.benchmark} | Stock: {stats.stock}</span>"
    )

    theme.apply_layout(fig, title=full_title, x_title="Route Length", y_title="Percentage (%)", legend_top=True)

    fig.update_layout(barmode="group", yaxis=dict(range=[0, 100]))
    return fig


def plot_comparison(models_stats: list[ModelStatistics], metric_type: str = "Top-K", k: int = 1) -> go.Figure:
    """
    Plots a direct comparison between multiple models for a specific metric.
    Uses jittered scatter plots with error bars.
    """
    # 1. Transform Data
    series_list = adapters.stats_to_comparison_series(models_stats, metric_type, k)

    # 2. Calculate Jitter/Offsets
    # We need to shift X values slightly so error bars don't overlap
    offsets = _calculate_offsets(len(series_list), width=0.6)

    # 3. Render
    fig = go.Figure()

    # Collect all x-values to force integer ticks later
    all_x = set()

    for i, series in enumerate(series_list):
        all_x.update(series.x)
        _render_series(fig, series, x_offset=offsets[i])

    # 4. Apply Theme
    title_suffix = f"(k={k})" if metric_type == "Top-K" else ""

    theme.apply_layout(
        fig,
        title=f"Model Comparison: {metric_type} {title_suffix}",
        x_title="Route Difficulty (Depth)",
        y_title="Percentage (%)",
    )

    # Force integer ticks for Depth
    if all_x:
        sorted_x = sorted([x for x in all_x if isinstance(x, (int, float))])
        fig.update_xaxes(tickmode="array", tickvals=sorted_x, ticktext=[f"Depth {int(x)}" for x in sorted_x])

    fig.update_yaxes(range=[0, 100])
    return fig


def plot_overall_summary(models_stats: list[ModelStatistics]) -> go.Figure:
    """
    Plots a high-level summary comparing Overall performance across key metrics.
    """
    # 1. Transform
    series_list = adapters.stats_to_overall_series(models_stats)

    # 2. Offsets
    offsets = _calculate_offsets(len(series_list), width=0.6)

    # 3. Render
    fig = go.Figure()
    for i, series in enumerate(series_list):
        _render_series(fig, series, x_offset=offsets[i])

    # 4. Theme
    # Map the integer indices back to labels for the X-axis
    # (This mirrors the order in adapters.stats_to_overall_series)
    labels = ["Solvability", "Top-1", "Top-5", "Top-10"]

    theme.apply_layout(fig, title="Overall Performance Summary", x_title="Metric", y_title="Percentage (%)")

    fig.update_xaxes(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=labels)
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_performance_matrix(models_stats: list[ModelStatistics]) -> go.Figure:
    """
    Creates a comprehensive heatmap: Models (Columns) vs Metrics (Rows).
    """
    # 1. Transform
    data = adapters.stats_to_heatmap_matrix(models_stats)

    # 2. Render
    fig = go.Figure(
        data=go.Heatmap(
            z=data.z,
            x=data.x_labels,
            y=data.y_labels,
            text=data.text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            xgap=1,
            ygap=1,
            hovertemplate="<b>%{x}</b><br>%{y}: %{z:.1f}%<extra></extra>",
        )
    )

    theme.apply_layout(fig, title=data.title, x_title="Model", y_title="Metric", legend_top=False)
    return fig


def plot_ranking(ranking: list[RankResult], metric_name: str) -> go.Figure:
    """
    Plots probabilistic ranking heatmap.
    """
    # 1. Prepare Data (Logic kept locally as it's simple/specific)
    y_labels = [r.model_name for r in ranking][::-1]  # Best on top
    n_models = len(ranking)
    x_labels = [f"Rank {i}" for i in range(1, n_models + 1)]

    z_values = []
    text_values = []

    for r in ranking[::-1]:
        row_z = []
        row_t = []
        for rank in range(1, n_models + 1):
            prob = r.rank_probs.get(rank, 0.0)
            row_z.append(prob)
            # Only show text if > 1%
            txt = f"{prob:.0%}" if prob > 0.01 else ""
            row_t.append(txt)
        z_values.append(row_z)
        text_values.append(row_t)

    # 2. Render
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


def plot_pairwise_matrix(comparisons: list[ModelComparison], metric_name: str) -> go.Figure:
    """
    Plots a Win/Loss matrix (Model A vs Model B).
    """
    # 1. Extract unique models
    models = sorted(list(set([c.model_a for c in comparisons])))
    model_map = {m: i for i, m in enumerate(models)}
    n = len(models)

    # 2. Initialize Matrix
    z_values = [[None] * n for _ in range(n)]
    text_values = [[""] * n for _ in range(n)]
    max_diff = 0.0

    for c in comparisons:
        row = model_map[c.model_a]
        col = model_map[c.model_b]

        # Difference A - B
        diff = c.diff_mean
        z_values[row][col] = diff
        max_diff = max(max_diff, abs(diff))

        sig_mark = "★" if c.is_significant else ""
        text_values[row][col] = f"{diff:+.1%}{sig_mark}"

    # 3. Reverse for Y-axis (Top-to-Bottom)
    models_y = models[::-1]
    z_values = z_values[::-1]
    text_values = text_values[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=models,
            y=models_y,
            text=text_values,
            texttemplate="%{text}",
            colorscale="RdBu",  # Red (loss) to Blue (win)
            zmid=0,
            zmin=-max_diff,
            zmax=max_diff,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y} vs %{x}</b><br>Diff: %{z:+.1%}<extra></extra>",
        )
    )

    # Auto-scale height based on number of models
    height = 600 + (n * 20)

    theme.apply_layout(
        fig,
        title=f"Pairwise Comparison Matrix: {metric_name}",
        x_title="Opponent",
        y_title="Model (Row - Col)",
        height=height,
        legend_top=False,
    )
    return fig


# --- Internal Rendering Helpers ---


def _render_series(fig: go.Figure, series: PlotSeries, x_offset: float = 0.0) -> None:
    """
    Unified renderer for a data series.
    Handles Scatter vs Bar, error bars, offsets, and hover data.
    """

    # Apply Offset to X if strictly numeric
    x_data = series.x
    if x_offset != 0.0 and series.x and isinstance(series.x[0], (int, float)):
        x_data = [x + x_offset for x in series.x]

    # Construct Hover Template
    # Expects custom_data in format: [N, CI_Low, CI_High, Reliability, optional_label]
    hover_tmpl = (
        f"<b>{series.name}</b><br>"
        "Value: %{y:.1f}%<br>"
        "N=%{customdata[0]}<br>"
        "CI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
        "Status: %{customdata[3]}"
        "<extra></extra>"
    )

    # Common args
    common_args = dict(
        name=series.name,
        x=x_data,
        y=series.y,
        marker_color=series.color,
        customdata=series.custom_data,
        hovertemplate=hover_tmpl,
    )

    # Error Bars
    error_y = None
    if series.y_err_upper:
        error_y = dict(
            type="data",
            symmetric=False,
            array=series.y_err_upper,
            arrayminus=series.y_err_lower,
            visible=True,
            width=4 if series.mode_hint == "scatter" else None,
            thickness=2,
        )

    # Dispatch based on hint
    if series.mode_hint == "bar":
        fig.add_trace(go.Bar(**common_args, error_y=error_y))

    elif series.mode_hint == "scatter":
        fig.add_trace(
            go.Scatter(
                **common_args,
                mode="markers" if x_offset != 0 else "markers+lines",
                error_y=error_y,
                marker=dict(color=series.color, size=10, symbol="circle"),
            )
        )


def _calculate_offsets(n_items: int, width: float = 0.6) -> list[float]:
    """
    Calculates X-axis offsets to center a cluster of `n_items`
    within a given `width`.
    """
    if n_items <= 1:
        return [0.0]

    step = width / max(1, (n_items - 1))
    start = -width / 2

    return [start + (i * step) for i in range(n_items)]
