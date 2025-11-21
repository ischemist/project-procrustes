"""
Plot generation functions.

This module handles the rendering of data into Plotly figures.
It relies on `retrocast.visualization.adapters` for data transformation
and `retrocast.visualization.theme` for styling.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retrocast.models.stats import ModelComparison, ModelStatistics, RankResult
from retrocast.visualization import adapters, theme
from retrocast.visualization.adapters import PlotSeries

# --- Main Plotting Functions ---


def plot_diagnostics(stats: ModelStatistics) -> go.Figure:
    """
    Plots performance diagnostics for a single model (Solvability & Top-K vs Length).
    """
    series_list = adapters.stats_to_diagnostic_series(stats)
    fig = go.Figure()

    for series in series_list:
        _render_series(fig, series)

    full_title = (
        f"<b>Performance Diagnostics: {stats.model_name}</b><br>"
        f"<span style='font-size: 12px;'>Benchmark: {stats.benchmark} | Stock: {stats.stock}</span>"
    )
    theme.apply_layout(fig, title=full_title, x_title="Route Length", y_title="Percentage (%)")
    fig.update_layout(barmode="group", yaxis_range=[0, 100])
    return fig


def plot_comparison(models_stats: list[ModelStatistics], metric_type: str = "Top-K", k: int = 1) -> go.Figure:
    """
    Plots a direct comparison between multiple models for a specific metric.
    """
    series_list = adapters.stats_to_comparison_series(models_stats, metric_type, k)
    offsets = _calculate_offsets(len(series_list), width=0.6)
    fig = go.Figure()
    all_x = set()

    for i, series in enumerate(series_list):
        all_x.update(series.x)
        _render_series(fig, series, x_offset=offsets[i])

    title_suffix = f"(k={k})" if metric_type == "Top-K" else ""
    theme.apply_layout(
        fig,
        title=f"Model Comparison: {metric_type} {title_suffix}",
        x_title="Route Difficulty (Length)",
        y_title="Percentage (%)",
    )

    if all_x:
        sorted_x = sorted(list(all_x))
        # FIX: Changed "Depth" to "Length" in tick labels
        fig.update_xaxes(tickmode="array", tickvals=sorted_x, ticktext=[f"Length {int(x)}" for x in sorted_x])
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_overall_summary(models_stats: list[ModelStatistics], top_k_values: list[int] | None = None) -> go.Figure:
    """
    Plots a high-level summary comparing Overall performance across key metrics.

    Args:
        models_stats: List of model statistics
        top_k_values: List of k values to display (default: [1, 2, 3, 4, 5, 10, 20, 50])
    """
    if top_k_values is None:
        top_k_values = [1, 2, 3, 4, 5, 10, 20, 50]

    series_list = adapters.stats_to_overall_series(models_stats, top_k_values)
    offsets = _calculate_offsets(len(series_list), width=0.6)
    fig = go.Figure()

    for i, series in enumerate(series_list):
        _render_series(fig, series, x_offset=offsets[i])

    labels = ["Solvability"] + [f"Top-{k}" for k in top_k_values]
    theme.apply_layout(fig, title="Overall Performance Summary", x_title="Metric", y_title="Percentage (%)")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels)
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_performance_matrix(models_stats: list[ModelStatistics]) -> go.Figure:
    """
    Creates a split heatmap with separate color scales for Solvability and Top-K.
    """
    data = adapters.stats_to_heatmap_matrix(models_stats)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.10, 0.9],
        horizontal_spacing=0.02,
        subplot_titles=(data.solvability.title, data.top_k.title),
    )

    # --- Panel 1: Solvability ---
    solv_data = data.solvability
    fig.add_trace(
        go.Heatmap(
            z=solv_data.z,
            x=solv_data.x_labels,
            y=solv_data.y_labels,
            text=solv_data.text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            coloraxis="coloraxis1",
        ),
        row=1,
        col=1,
    )

    # --- Panel 2: Top-K Accuracy ---
    top_k_data = data.top_k
    fig.add_trace(
        go.Heatmap(
            z=top_k_data.z,
            x=top_k_data.x_labels,
            y=top_k_data.y_labels,
            text=top_k_data.text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            coloraxis="coloraxis2",
        ),
        row=1,
        col=2,
    )

    # --- Apply Theme & Final Layout ---
    # We now configure the color axes inside the main layout update
    theme.apply_layout(fig, legend_top=False, height=400 + (len(data.solvability.y_labels) * 20), width=1000)

    # FIX: Explicitly configure and hide each color axis
    fig.update_layout(
        # This defines the properties for our custom color axes
        coloraxis1=dict(colorscale="Greens", showscale=False, cmin=50, cmax=100),
        coloraxis2=dict(
            colorscale="Blues",
            showscale=False,
            cmin=0,
            cmax=max(v for row in top_k_data.z for v in row if v is not None) or 100,
        ),
    )

    fig.update_yaxes(title_text="Model", row=1, col=1)

    return fig


def plot_ranking(ranking: list[RankResult], metric_name: str) -> go.Figure:
    """Plots probabilistic ranking heatmap."""
    y_labels = [r.model_name for r in ranking][::-1]
    n_models = len(ranking)
    x_labels = [f"Rank {i}" for i in range(1, n_models + 1)]
    z_values, text_values = [], []

    for r in ranking[::-1]:
        row_z, row_t = [], []
        for rank in range(1, n_models + 1):
            prob = r.rank_probs.get(rank, 0.0)
            row_z.append(prob)
            row_t.append(f"{prob:.0%}" if prob > 0.01 else "")
        z_values.append(row_z)
        text_values.append(row_t)

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
    """Plots a Win/Loss matrix (Model A vs Model B)."""
    models = sorted(list(set(c.model_a for c in comparisons) | set(c.model_b for c in comparisons)))
    model_map = {m: i for i, m in enumerate(models)}
    n = len(models)
    z_values, text_values = [[None] * n for _ in range(n)], [[""] * n for _ in range(n)]
    max_diff = 0.0

    for c in comparisons:
        row, col = model_map[c.model_a], model_map[c.model_b]
        diff = c.diff_mean * 100  # Convert to percentage points
        z_values[row][col], z_values[col][row] = diff, -diff
        max_diff = max(max_diff, abs(diff))
        sig_a = "★" if c.is_significant else ""
        sig_b = "★" if c.is_significant else ""
        text_values[row][col] = f"{diff:+.1f}%{sig_a}"
        text_values[col][row] = f"{-diff:+.1f}%{sig_b}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=models,
            y=models,
            text=text_values,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-max_diff,
            zmax=max_diff,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y} vs %{x}</b><br>Diff (Row-Col): %{z:+.1f}%<extra></extra>",
        )
    )
    height = 500 + (n * 25)
    theme.apply_layout(
        fig, title=f"Pairwise Comparison Matrix (Row - Col): {metric_name}", height=height, legend_top=False
    )
    return fig


# --- Internal Rendering Helpers ---


def _render_series(fig: go.Figure, series: PlotSeries, x_offset: float = 0.0):
    """Unified renderer for a data series."""
    x_data = (
        [x + x_offset for x in series.x]
        if x_offset != 0.0 and series.x and isinstance(series.x[0], (int, float))
        else series.x
    )

    hover_tmpl = (
        f"<b>{series.name}</b><br>"
        + ("Value" if series.mode_hint == "bar" else "Metric")
        + ": %{y:.1f}%<br>"
        + "N=%{customdata[0]}<br>"
        + "CI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
        + "Status: %{customdata[3]}"
        + "<extra></extra>"
    )

    common_args = dict(
        name=series.name,
        x=x_data,
        y=series.y,
        marker_color=series.color,
        customdata=series.custom_data,
        hovertemplate=hover_tmpl,
    )
    error_y = (
        dict(type="data", symmetric=False, array=series.y_err_upper, arrayminus=series.y_err_lower, visible=True)
        if series.y_err_upper
        else None
    )

    if series.mode_hint == "bar":
        fig.add_trace(go.Bar(**common_args, error_y=error_y))
    elif series.mode_hint == "scatter":
        # FIX: No more conditional lines. Always markers for comparison plots.
        fig.add_trace(
            go.Scatter(
                **common_args,
                mode="markers",
                error_y={**error_y, "width": 4, "thickness": 1.5},
                marker=dict(color=series.color, size=10, symbol="circle"),
            )
        )


def _calculate_offsets(n_items: int, width: float = 0.6) -> list[float]:
    """Calculates X-axis offsets to center a cluster."""
    if n_items <= 1:
        return [0.0]
    step = width / (n_items - 1)
    return [-width / 2 + (i * step) for i in range(n_items)]
