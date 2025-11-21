"""
Data adapters for visualization.

This module transforms rich Pydantic models (ModelStatistics) into
simple data structures (PlotSeries, HeatmapData) ready for plotting.
It strictly enforces a separation between logic and rendering.
"""

from dataclasses import dataclass

from retrocast.models.stats import ModelStatistics
from retrocast.visualization.theme import get_metric_color, get_model_color


@dataclass
class PlotSeries:
    """
    A generic container for 1D plotting data (Scatter, Bar, Line).
    Decouples the plotting logic from the source data structure.
    """

    name: str
    x: list[int | float | str]
    y: list[float]
    # Error bars (deltas relative to y, not absolute values)
    y_err_upper: list[float] | None = None
    y_err_lower: list[float] | None = None
    color: str | None = None
    # List of list of values for hover template
    custom_data: list[list] | None = None
    # Type hint for the renderer (e.g. 'bar', 'scatter')
    mode_hint: str = "scatter"


@dataclass
class HeatmapData:
    """
    Container for 2D Matrix data.
    """

    z: list[list[float | None]]  # The matrix
    x_labels: list[str]
    y_labels: list[str]
    text: list[list[str]]  # Text to display in cells
    title: str


# --- Transformers ---


def stats_to_diagnostic_series(stats: ModelStatistics) -> list[PlotSeries]:
    """
    Converts a single model's stats into series for a diagnostic plot.
    X-axis: Route Length (Depth)
    Series: Solvability, Top-1, Top-5, Top-10
    """
    series_list = []

    # 1. Solvability
    series_list.append(
        _create_depth_series(stats.solvability, name="Solvability", color=get_metric_color("solvability"), mode="bar")
    )

    # 2. Top-K Accuracies
    # Sort keys to ensure logical legend order
    for k in sorted(stats.top_k_accuracy.keys()):
        # We typically only plot k=1, 5, 10 to avoid clutter
        if k in [1, 5, 10]:
            series_list.append(
                _create_depth_series(
                    stats.top_k_accuracy[k], name=f"Top-{k}", color=get_metric_color("top", k), mode="bar"
                )
            )

    return series_list


def stats_to_comparison_series(models_stats: list[ModelStatistics], metric_type: str, k: int = 1) -> list[PlotSeries]:
    """
    Converts multiple models into series for a direct comparison on a specific metric.
    X-axis: Route Length (Depth)
    Series: One per Model
    """
    series_list = []

    for stats in models_stats:
        # Determine which metric object to grab
        if metric_type == "Solvability":
            metric_obj = stats.solvability
        elif metric_type == "Top-K":
            metric_obj = stats.top_k_accuracy.get(k)
        else:
            metric_obj = None

        if not metric_obj:
            continue

        series_list.append(
            _create_depth_series(
                metric_obj,
                name=stats.model_name,
                color=get_model_color(stats.model_name),
                mode="scatter",  # Comparisons are usually scatter/line
            )
        )

    return series_list


def stats_to_overall_series(models_stats: list[ModelStatistics]) -> list[PlotSeries]:
    """
    Converts multiple models into series for Overall performance summary.
    X-axis: Metric Name (Solvability, Top-1, etc.)
    Series: One per Model
    """
    metrics_config = [
        {"key": "solvability", "label": "Solvability"},
        {"key": "top-1", "label": "Top-1"},
        {"key": "top-5", "label": "Top-5"},
        {"key": "top-10", "label": "Top-10"},
    ]

    series_list = []

    for stats in models_stats:
        x_vals = []
        y_vals = []
        y_up = []
        y_low = []
        custom = []

        for i, config in enumerate(metrics_config):
            key = config["key"]
            res = None

            if key == "solvability":
                res = stats.solvability.overall
            elif key.startswith("top-"):
                k = int(key.split("-")[1])
                if k in stats.top_k_accuracy:
                    res = stats.top_k_accuracy[k].overall

            if res:
                x_vals.append(i)  # Integer index for X
                y_vals.append(res.value * 100)
                y_up.append((res.ci_upper - res.value) * 100)
                y_low.append((res.value - res.ci_lower) * 100)

                custom.append(
                    [res.n_samples, res.ci_lower * 100, res.ci_upper * 100, res.reliability.code, config["label"]]
                )

        series_list.append(
            PlotSeries(
                name=stats.model_name,
                x=x_vals,
                y=y_vals,
                y_err_upper=y_up,
                y_err_lower=y_low,
                color=get_model_color(stats.model_name),
                custom_data=custom,
                mode_hint="scatter",
            )
        )

    return series_list


def stats_to_heatmap_matrix(models_stats: list[ModelStatistics]) -> HeatmapData:
    """
    Creates a comprehensive matrix: Models (Columns) vs Metrics (Rows).
    Metrics: Solvability, Top-1, 2, 3, 4, 5, 10, 20, 50 (if available).
    """
    # 1. Define Metrics (Rows)
    # Always start with Solvability
    metric_keys = ["Solvability"]
    # Find all unique K present across all models
    all_k = set()
    for m in models_stats:
        all_k.update(m.top_k_accuracy.keys())

    sorted_k = sorted(list(all_k))
    metric_keys.extend([f"Top-{k}" for k in sorted_k])

    # 2. Define Models (Columns)
    model_names = [m.model_name for m in models_stats]

    # 3. Build Matrix
    z_values = []  # List of rows
    text_values = []

    for metric_label in metric_keys:
        z_row = []
        t_row = []

        for stats in models_stats:
            val = None

            if metric_label == "Solvability":
                val = stats.solvability.overall.value
            else:
                k = int(metric_label.split("-")[1])
                if k in stats.top_k_accuracy:
                    val = stats.top_k_accuracy[k].overall.value

            if val is not None:
                z_row.append(val * 100)  # Store as percentage
                t_row.append(f"{val:.1%}")
            else:
                z_row.append(None)
                t_row.append("")

        z_values.append(z_row)
        text_values.append(t_row)

    # Note: We structure it so Y=Metrics, X=Models.
    # Plotly Heatmap expects z[row][col].

    return HeatmapData(
        z=z_values,
        x_labels=model_names,
        y_labels=metric_keys,
        text=text_values,
        title="Comprehensive Performance Matrix",
    )


# --- Internal Helper ---


def _create_depth_series(
    metric_obj,  # StratifiedMetric
    name: str,
    color: str,
    mode: str,
) -> PlotSeries:
    """Helper to extract stratified data into a PlotSeries."""
    sorted_keys = sorted(metric_obj.by_group.keys())

    x_vals = []
    y_vals = []
    y_up = []
    y_low = []
    custom = []

    for k in sorted_keys:
        res = metric_obj.by_group[k]

        x_vals.append(k)
        y_vals.append(res.value * 100)
        y_up.append((res.ci_upper - res.value) * 100)
        y_low.append((res.value - res.ci_lower) * 100)

        custom.append([res.n_samples, res.ci_lower * 100, res.ci_upper * 100, res.reliability.code])

    return PlotSeries(
        name=name,
        x=x_vals,
        y=y_vals,
        y_err_upper=y_up,
        y_err_lower=y_low,
        color=color,
        custom_data=custom,
        mode_hint=mode,
    )
