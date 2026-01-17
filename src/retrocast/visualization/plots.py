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
    """
    Plots a Win/Loss matrix (Model A vs Model B).
    Row (Y) - Col (X).
    """
    # 1. Extract and Sort Unique Models
    # We sort them to ensure the matrix indices match the labels perfectly
    models = sorted(list(set([c.model_a for c in comparisons])))
    model_map = {m: i for i, m in enumerate(models)}
    n = len(models)

    # 2. Initialize Matrix Data Containers
    z_values = [[None] * n for _ in range(n)]
    text_values = [[""] * n for _ in range(n)]
    custom_data = [[None] * n for _ in range(n)]  # For rich hover info

    max_diff = 0.0

    for c in comparisons:
        row = model_map[c.model_a]
        col = model_map[c.model_b]

        # Value: Difference (Row - Col)
        diff = c.diff_mean
        z_values[row][col] = diff
        max_diff = max(max_diff, abs(diff))

        # Cell Text: "+5.2% ★"
        sig_mark = "★" if c.is_significant else ""
        text_values[row][col] = f"{diff:+.1%}{sig_mark}"

        # Rich Hover Data construction
        # [Row Model, Col Model, Significance Label, Winner/Description]
        sig_str = "Significant ✅" if c.is_significant else "Insignificant ⚠️"

        if diff > 0:
            desc = f"<b>{c.model_a}</b> beats {c.model_b}"
        else:
            desc = f"<b>{c.model_b}</b> beats {c.model_a}"

        custom_data[row][col] = [
            c.model_a,  # 0: Row Name
            c.model_b,  # 1: Col Name
            sig_str,  # 2: Sig Status
            desc,  # 3: Description
        ]

    # 3. Render Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=models,
            y=models,  # Pass in normal order, we flip axis in layout
            text=text_values,
            texttemplate="%{text}",
            customdata=custom_data,
            colorscale="RdBu",  # Red (loss) to Blue (win)
            zmid=0,
            zmin=-max_diff,
            zmax=max_diff,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "<b>Matchup:</b> %{customdata[0]} vs %{customdata[1]}<br>"
                "<b>Difference:</b> %{z:+.2%}<br>"
                "<b>Result:</b> %{customdata[3]}<br>"
                "<b>Status:</b> %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    # Auto-scale height based on number of models
    height = 600 + (n * 30)
    width = 700 + (n * 30)

    theme.apply_layout(
        fig,
        title=f"Pairwise Comparison Matrix: {metric_name}",
        x_title="Opponent (Column)",
        y_title="Model (Row)",
        height=height,
        width=width,
        legend_top=False,
    )

    # This makes the matrix read like a table (Top-to-Bottom).
    fig.update_yaxes(autorange="reversed")

    return fig


def plot_stability_analysis(data_list: list[adapters.StabilityData], bench_name: str, model_name: str) -> go.Figure:
    """
    Plots a Forest Plot showing variance across random seeds.
    """
    fig = go.Figure()

    for data in data_list:
        # 1. The Scatter Points with Error Bars
        fig.add_trace(
            go.Scatter(
                x=data.values,
                y=[f"Seed {s}" for s in data.seeds],
                name=data.metric_name,
                mode="markers",
                marker=dict(color=data.color, size=8),
                error_x=dict(type="data", array=data.errors_plus, arrayminus=data.errors_minus, visible=True),
                hovertemplate=(f"<b>{data.metric_name}</b><br>Seed: %{{y}}<br>Value: %{{x:.2f}}%<br><extra></extra>"),
            )
        )

        # 2. The Grand Mean Line (Vertical)
        fig.add_vline(
            x=data.grand_mean,
            line_width=2,
            line_dash="dot",
            line_color=data.color,
            annotation_text=f"μ={data.grand_mean:.1f}% (σ={data.std_dev:.2f})",
            annotation_position="top right",
        )
    fig.update_xaxes(dtick=10, range=[20, 100])
    theme.apply_layout(
        fig,
        title=f"[{bench_name}] Stability Analysis ({model_name})",
        x_title="Performance (%)",
        y_title="Seed Variant",
        height=max(600, len(data_list[0].seeds) * 25),  # Dynamic height
        legend_top=True,
    )

    return fig


def plot_pareto_frontier(
    models_stats: list[ModelStatistics],
    model_config: dict[str, dict[str, str]],
    hourly_costs: dict[str, float],
    k: int = 10,
    time_based: bool = False,
) -> go.Figure:
    """
    Creates a Pareto frontier plot showing cost vs accuracy trade-offs.

    Args:
        models_stats: List of model statistics
        model_config: Dict mapping model_name -> {legend, short, color}
        hourly_costs: Dict mapping model_name -> hourly compute cost (USD)
        k: Which top-k accuracy to plot (default: 10)
        time_based: If True, use wall time (minutes) instead of cost (USD) for X-axis

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Extract benchmark name for title (all stats should have same benchmark)
    # benchmark = models_stats[0].benchmark if models_stats else "Unknown"
    # stock = models_stats[0].stock if models_stats else "Unknown"

    # Collect all points for Pareto frontier calculation
    pareto_points = []

    for stats in models_stats:
        model_name = stats.model_name

        # Skip if no wall time data
        if stats.total_wall_time is None:
            continue

        # Skip if no cost data (only required in cost mode)
        if not time_based and model_name not in hourly_costs:
            continue

        # Skip if no top-k data
        if k not in stats.top_k_accuracy:
            continue

        # Calculate X-axis value: wall time (minutes) or cost (USD)
        wall_time_minutes = stats.total_wall_time / 60
        wall_time_hours = stats.total_wall_time / 3600

        if time_based:
            x_value = wall_time_minutes
        else:
            x_value = wall_time_hours * hourly_costs[model_name]

        # Get accuracy metrics
        metric = stats.top_k_accuracy[k].overall
        accuracy = metric.value * 100
        ci_lower = metric.ci_lower * 100
        ci_upper = metric.ci_upper * 100

        # Calculate error bar lengths
        error_minus = accuracy - ci_lower
        error_plus = ci_upper - accuracy

        # Get display config (with fallback)
        config = model_config.get(
            model_name,
            {
                "legend": model_name,
                "short": model_name[:10],
                "color": theme.get_model_color(model_name),
            },
        )

        # Store point for Pareto frontier
        pareto_points.append((x_value, accuracy, model_name))

        # Build hover template based on mode
        if time_based:
            hover_template = (
                "<b>%{customdata[0]}</b><br>"
                "Wall Time: %{customdata[1]:.1f} min<br>"
                f"Top-{k} Accuracy: %{{y:.1f}}%<br>"
                "CI: [%{customdata[4]:.1f}%, %{customdata[5]:.1f}%]<br>"
                "N=%{customdata[3]}<br>"
                "Status: %{customdata[6]}"
                "<extra></extra>"
            )
        else:
            hover_template = (
                "<b>%{customdata[0]}</b><br>"
                "Cost: $%{customdata[1]:.2f}<br>"
                "Wall Time: %{customdata[2]:.1f} min<br>"
                f"Top-{k} Accuracy: %{{y:.1f}}%<br>"
                "CI: [%{customdata[4]:.1f}%, %{customdata[5]:.1f}%]<br>"
                "N=%{customdata[3]}<br>"
                "Status: %{customdata[6]}"
                "<extra></extra>"
            )

        # Add scatter point with error bars
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[accuracy],
                name=config["legend"],
                mode="markers+text",
                marker=dict(
                    color=config["color"],
                    size=12,
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
                text=[config["short"]],
                textposition="middle right",
                textfont=dict(size=14),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[error_plus],
                    arrayminus=[error_minus],
                    visible=True,
                    width=4,
                    thickness=1.5,
                ),
                customdata=[
                    [
                        model_name,
                        x_value,
                        wall_time_minutes,
                        metric.n_samples,
                        ci_lower,
                        ci_upper,
                        metric.reliability.code,
                    ]
                ],
                hovertemplate=hover_template,
            )
        )

    # Calculate and draw Pareto frontier
    if pareto_points:
        # Sort by cost (ascending)
        pareto_points.sort(key=lambda p: p[0])

        # Find Pareto-optimal points (no point has both lower cost AND higher accuracy)
        pareto_optimal = []
        max_accuracy_so_far = -float("inf")

        for cost, accuracy, _model_name in pareto_points:
            if accuracy > max_accuracy_so_far:
                pareto_optimal.append((cost, accuracy))
                max_accuracy_so_far = accuracy

        # Draw the Pareto frontier line
        if len(pareto_optimal) > 1:
            pareto_x = [p[0] for p in pareto_optimal]
            pareto_y = [p[1] for p in pareto_optimal]

            fig.add_trace(
                go.Scatter(
                    x=pareto_x,
                    y=pareto_y,
                    mode="lines",
                    name="Pareto Frontier",
                    line=dict(color="rgba(128,128,128,0.5)", width=2, dash="dash"),
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    # Apply layout
    # title = f"<b>Pareto Frontier: Cost vs Accuracy</b><br><span style='font-size: 12px;'>Benchmark: {benchmark} | Stock: {stock}</span>"
    x_title = "Wall Time (minutes)" if time_based else "Total Cost (USD)"
    theme.apply_layout(
        fig,
        # title=title,
        x_title=x_title,
        y_title=f"Top-{k} Accuracy (%)",
        height=600,
        width=1200,
    )

    fig.update_yaxes(range=[0, 100])
    # fig.update_xaxes(range=[-0.2, 3.2])
    fig.update_layout(legend=dict(y=0.9, orientation="h"))

    return fig


# --- Internal Rendering Helpers ---


def _render_series(fig: go.Figure, series: PlotSeries, x_offset: float = 0.0):
    """Unified renderer for a data series."""
    if x_offset != 0.0 and series.x:
        # Type narrowing: cast to numeric list for type checker
        numeric_x: list[int | float] = [x for x in series.x if isinstance(x, (int, float))]
        x_data = [x + x_offset for x in numeric_x]
    else:
        # No offset, x values can be any type (int, float, or str)
        x_data = series.x

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
