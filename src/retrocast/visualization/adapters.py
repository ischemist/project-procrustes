"""
Data adapters for visualization.

This module transforms rich Pydantic models (ModelStatistics) into
simple data structures (PlotSeries, HeatmapData) ready for plotting.
It strictly enforces a separation between logic and rendering.
"""

from dataclasses import dataclass
from typing import Any

from retrocast.models.stats import ModelStatistics, StratifiedMetric
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
    """Container for 2D Matrix data."""

    z: list[list[float | None]]
    x_labels: list[str]
    y_labels: list[str]
    text: list[list[str]]
    title: str


@dataclass
class SplitHeatmapData:
    """Container for splitting a heatmap into multiple panels."""

    solvability: HeatmapData
    top_k: HeatmapData


@dataclass
class StabilityData:
    """Container for seed stability analysis."""

    metric_name: str
    seeds: list[str]
    values: list[float]  # Percentages
    errors_plus: list[float]  # CI Upper - Value
    errors_minus: list[float]  # Value - CI Lower
    grand_mean: float
    std_dev: float
    color: str


# --- Transformers ---


def stats_to_diagnostic_series(stats: ModelStatistics) -> list[PlotSeries]:
    """
    Converts a single model's stats into series for a diagnostic plot.
    X-axis: Route Length
    Series: Solvability, Top-1, Top-5, Top-10
    """
    series_list = []

    # 1. Solvability
    series_list.append(
        _create_length_series(stats.solvability, name="Solvability", color=get_metric_color("solvability"), mode="bar")
    )

    # 2. Top-K Accuracies
    for k in sorted(stats.top_k_accuracy.keys()):
        if k in [1, 2, 3, 4, 5, 10, 20, 50]:
            series_list.append(
                _create_length_series(
                    stats.top_k_accuracy[k], name=f"Top-{k}", color=get_metric_color("top", k), mode="bar"
                )
            )

    return series_list


def stats_to_comparison_series(models_stats: list[ModelStatistics], metric_type: str, k: int = 1) -> list[PlotSeries]:
    """
    Converts multiple models into series for a direct comparison on a specific metric.
    X-axis: Route Length
    Series: One per Model
    """
    series_list = []

    for stats in models_stats:
        metric_obj = None
        if metric_type == "Solvability":
            metric_obj = stats.solvability
        elif metric_type == "Top-K":
            metric_obj = stats.top_k_accuracy.get(k)

        if metric_obj:
            series_list.append(
                _create_length_series(
                    metric_obj, name=stats.model_name, color=get_model_color(stats.model_name), mode="scatter"
                )
            )

    return series_list


def stats_to_overall_series(
    models_stats: list[ModelStatistics], top_k_values: list[int] | None = None
) -> list[PlotSeries]:
    """
    Converts multiple models into series for Overall performance summary.
    X-axis: Metric Name (Solvability, Top-1, etc.)
    Series: One per Model

    Args:
        models_stats: List of model statistics
        top_k_values: List of k values to display (default: [1, 2, 3, 4, 5, 10, 20, 50])
    """
    if top_k_values is None:
        top_k_values = [1, 2, 3, 4, 5, 10, 20, 50]

    metrics_config = [{"key": "solvability", "label": "Solvability"}]
    metrics_config.extend([{"key": f"top-{k}", "label": f"Top-{k}"} for k in top_k_values])
    series_list = []

    for stats in models_stats:
        x_vals, y_vals, y_up, y_low, custom = [], [], [], [], []
        for i, config in enumerate(metrics_config):
            res = None
            if config["key"] == "solvability":
                res = stats.solvability.overall
            elif config["key"].startswith("top-"):
                k = int(config["key"].split("-")[1])
                if k in stats.top_k_accuracy:
                    res = stats.top_k_accuracy[k].overall

            if res:
                x_vals.append(i)
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


def stats_to_heatmap_matrix(models_stats: list[ModelStatistics]) -> SplitHeatmapData:
    """
    Creates data for a split heatmap:
    - Panel 1: Solvability
    - Panel 2: Top-K Accuracies
    """
    # 1. Define Metrics for each panel
    solv_metrics = ["Solvability"]
    all_k = set()
    for m in models_stats:
        all_k.update(m.top_k_accuracy.keys())
    top_k_metrics = [f"Top-{k}" for k in sorted(list(all_k))]

    # 2. Define Models (Rows / Y-axis) - shared for both
    model_names = [m.model_name for m in models_stats]

    # 3. Build Matrices
    solv_z, solv_text = [], []
    top_k_z, top_k_text = [], []

    for stats in models_stats:  # Outer loop for rows (models)
        # Solvability Panel
        solv_val = stats.solvability.overall.value
        solv_z.append([solv_val * 100])
        solv_text.append([f"{solv_val:.1%}"])

        # Top-K Panel
        top_k_row_z, top_k_row_text = [], []
        for metric_label in top_k_metrics:
            k = int(metric_label.split("-")[1])
            res = stats.top_k_accuracy.get(k)
            val = res.overall.value if res else None

            if val is not None:
                top_k_row_z.append(val * 100)
                top_k_row_text.append(f"{val:.1%}")
            else:
                top_k_row_z.append(None)
                top_k_row_text.append("")
        top_k_z.append(top_k_row_z)
        top_k_text.append(top_k_row_text)

    return SplitHeatmapData(
        solvability=HeatmapData(
            z=solv_z, x_labels=solv_metrics, y_labels=model_names, text=solv_text, title="Solvability"
        ),
        top_k=HeatmapData(
            z=top_k_z, x_labels=top_k_metrics, y_labels=model_names, text=top_k_text, title="Top-K Accuracy"
        ),
    )


def stats_to_stability_data(results_map: dict[str, dict[str, Any]], metric_key: str, color: str) -> StabilityData:
    """
    Converts a map of {seed: {metric: Result}} into plotting data.
    """
    import numpy as np

    # Sort seeds numerically if possible
    seeds = sorted(results_map.keys(), key=lambda x: int(x) if x.isdigit() else x)

    vals = []
    e_plus = []
    e_minus = []

    for seed in seeds:
        res = results_map[seed][metric_key]
        v = res.value * 100
        vals.append(v)
        e_plus.append((res.ci_upper * 100) - v)
        e_minus.append(v - (res.ci_lower * 100))

    raw = np.array(vals)

    return StabilityData(
        metric_name=metric_key,
        seeds=seeds,
        values=vals,
        errors_plus=e_plus,
        errors_minus=e_minus,
        grand_mean=float(np.mean(raw)),
        std_dev=float(np.std(raw)),
        color=color,
    )


# --- Internal Helper ---


def _create_length_series(metric_obj: StratifiedMetric, name: str, color: str, mode: str) -> PlotSeries:
    """Helper to extract stratified data by route length into a PlotSeries."""
    # FIX: Cast keys to int before sorting to handle "10" vs "2" correctly.
    sorted_keys = sorted(metric_obj.by_group.keys(), key=int)

    x_vals, y_vals, y_up, y_low, custom = [], [], [], [], []
    for k in sorted_keys:
        res = metric_obj.by_group[k]
        x_vals.append(int(k))  # Ensure x is numeric for jittering
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
