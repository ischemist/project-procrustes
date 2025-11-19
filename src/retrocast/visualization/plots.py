import plotly.graph_objects as go
from ischemist.plotly import Styler

from retrocast.models.stats import ModelStatistics, StratifiedMetric


def _create_metric_trace(metric: StratifiedMetric, name: str, color: str) -> go.Bar:
    """
    Helper to create a single bar trace with error bars.
    """
    # Extract sorted data
    # Sort by depth (assuming keys are ints or sortable)
    sorted_keys = sorted(metric.by_group.keys())

    x_vals = [f"Length {k}" for k in sorted_keys]
    y_vals = [metric.by_group[k].value * 100 for k in sorted_keys]  # Convert to %

    # Error bars
    # Plotly expects the length of the error bar, not the absolute value
    ci_upper = [metric.by_group[k].ci_upper * 100 for k in sorted_keys]
    ci_lower = [metric.by_group[k].ci_lower * 100 for k in sorted_keys]

    error_plus = [upp - y for upp, y in zip(ci_upper, y_vals, strict=True)]
    error_minus = [y - low for y, low in zip(y_vals, ci_lower, strict=True)]

    custom_data = []
    for low, upp, n, rel in zip(
        ci_lower,
        ci_upper,
        [metric.by_group[k].n_samples for k in sorted_keys],
        [metric.by_group[k].reliability.code for k in sorted_keys],
        strict=True,
    ):
        custom_data.append([n, low, upp, rel])

    return go.Bar(
        name=name,
        x=x_vals,
        y=y_vals,
        marker_color=color,
        error_y=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, visible=True),
        customdata=custom_data,
        hovertemplate=(
            f"<b>{name}</b>: %{{y:.1f}}%<br>"
            + "N=%{customdata[0]}<br>"
            + "CI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
            + "Status: %{customdata[3]}"
            + "<extra></extra>"
        ),
    )


def plot_single_model_diagnostics(stats: ModelStatistics) -> go.Figure:
    """
    Creates a grouped bar chart showing performance metrics stratified by depth.
    Includes Solvability, Top-1, Top-5, Top-10.
    """
    fig = go.Figure()

    # 1. Solvability (The Baseline)
    fig.add_trace(
        _create_metric_trace(
            stats.solvability,
            name="Solvability",
            color="#b892ff",  # Strong Blue
        )
    )

    # 2. Top-K Accuracies
    # We choose a few key Ks to avoid clutter
    k_colors = {
        1: "#ffc2e2",  # Orange
        5: "#ff90b3",  # Green
        10: "#ef7a85",  # Purple
    }

    for k in [1, 5, 10]:
        if k in stats.top_k_accuracy:
            fig.add_trace(
                _create_metric_trace(stats.top_k_accuracy[k], name=f"Top-{k}", color=k_colors.get(k, "#95A5A6"))
            )

    # 3. Layout Polish
    full_title = (
        f"<b>Performance Diagnostics: {stats.model_name}</b><br>"
        f"<span style='font-size: 12px;'>Benchmark: {stats.benchmark} | Stock: {stats.stock}</span>"
    )
    fig.update_layout(
        title=full_title,
        yaxis=dict(range=[0, 100]),
        xaxis=dict(title="Route Length"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1000,
        height=500,
    )
    Styler().apply_style(fig)

    return fig
