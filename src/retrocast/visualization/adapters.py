from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrocast.models.analysis import MetricSummary
from retrocast.visualization.depth import depth_group_sort_key, depth_group_value
from retrocast.visualization.theme import get_metric_color, get_model_color


@dataclass(frozen=True, slots=True)
class PlotSeries:
    name: str
    x: list[int | float | str]
    y: list[float]
    y_err_upper: list[float] | None = None
    y_err_lower: list[float] | None = None
    color: str | None = None
    custom_data: list[list[Any]] | None = None
    mode_hint: str = "scatter"


@dataclass(frozen=True, slots=True)
class HeatmapData:
    z: list[list[float | None]]
    x_labels: list[str]
    y_labels: list[str]
    text: list[list[str]]
    title: str


@dataclass(frozen=True, slots=True)
class SplitHeatmapData:
    solvability: HeatmapData
    top_k: HeatmapData


@dataclass(frozen=True, slots=True)
class StabilityData:
    metric_name: str
    seeds: list[str]
    values: list[float]
    errors_plus: list[float]
    errors_minus: list[float]
    grand_mean: float
    std_dev: float
    color: str


def stats_to_diagnostic_series(stats: Any) -> list[PlotSeries]:
    series = [_create_depth_series(_solvability_metric(stats), "Solvability", get_metric_color("solvability"), "bar")]
    for k in sorted(getattr(stats, "top_k_accuracy", {})):
        if k in {1, 2, 3, 4, 5, 10, 20, 50}:
            series.append(
                _create_depth_series(
                    stats.top_k_accuracy[k],
                    f"Top-{k}",
                    get_metric_color("top", k),
                    "bar",
                )
            )
    return series


def stats_to_comparison_series(models_stats: list[Any], metric_type: str, k: int = 1) -> list[PlotSeries]:
    series = []
    for stats in models_stats:
        metric = (
            _solvability_metric(stats) if metric_type == "Solvability" else getattr(stats, "top_k_accuracy", {}).get(k)
        )
        if metric is None:
            continue
        series.append(_create_depth_series(metric, stats.model_name, get_model_color(stats.model_name), "scatter"))
    return series


def stats_to_overall_series(models_stats: list[Any], top_k_values: list[int] | None = None) -> list[PlotSeries]:
    top_k_values = top_k_values or [1, 2, 3, 4, 5, 10, 20, 50]
    series = []
    for stats in models_stats:
        x_values: list[int | float | str] = []
        y_values: list[float] = []
        y_upper: list[float] = []
        y_lower: list[float] = []
        custom: list[list[Any]] = []
        metrics: list[tuple[str, MetricSummary | None]] = [("Solvability", _overall_metric(_solvability_metric(stats)))]
        metrics.extend((f"Top-{k}", _overall_metric(getattr(stats, "top_k_accuracy", {}).get(k))) for k in top_k_values)
        for index, (label, metric) in enumerate(metrics):
            if metric is None:
                continue
            x_values.append(index)
            value = metric.value * 100
            x_ci_low, x_ci_high = _metric_ci(metric)
            y_values.append(value)
            y_upper.append(x_ci_high * 100 - value)
            y_lower.append(value - x_ci_low * 100)
            custom.append([metric.count, x_ci_low * 100, x_ci_high * 100, _reliability_code(metric), label])
        series.append(
            PlotSeries(
                name=stats.model_name,
                x=x_values,
                y=y_values,
                y_err_upper=y_upper,
                y_err_lower=y_lower,
                color=get_model_color(stats.model_name),
                custom_data=custom,
                mode_hint="scatter",
            )
        )
    return series


def stats_to_heatmap_matrix(models_stats: list[Any]) -> SplitHeatmapData:
    top_k_values = sorted({k for stats in models_stats for k in getattr(stats, "top_k_accuracy", {})})
    model_names = [stats.model_name for stats in models_stats]
    solv_z: list[list[float | None]] = []
    solv_text: list[list[str]] = []
    top_z: list[list[float | None]] = []
    top_text: list[list[str]] = []
    for stats in models_stats:
        solv = _overall_metric(_solvability_metric(stats))
        solv_z.append([solv.value * 100 if solv else None])
        solv_text.append([f"{solv.value:.1%}" if solv else ""])
        top_row: list[float | None] = []
        text_row: list[str] = []
        for k in top_k_values:
            metric = _overall_metric(getattr(stats, "top_k_accuracy", {}).get(k))
            top_row.append(metric.value * 100 if metric else None)
            text_row.append(f"{metric.value:.1%}" if metric else "")
        top_z.append(top_row)
        top_text.append(text_row)
    return SplitHeatmapData(
        solvability=HeatmapData(solv_z, ["Solvability"], model_names, solv_text, "Solvability"),
        top_k=HeatmapData(top_z, [f"Top-{k}" for k in top_k_values], model_names, top_text, "Top-K Accuracy"),
    )


def stats_to_stability_data(
    results_map: dict[str, dict[str, MetricSummary]], metric_key: str, color: str
) -> StabilityData:
    import numpy as np

    seeds = sorted(results_map, key=lambda seed: int(seed) if seed.isdigit() else seed)
    values: list[float] = []
    errors_plus: list[float] = []
    errors_minus: list[float] = []
    for seed in seeds:
        metric = results_map[seed][metric_key]
        ci_low, ci_high = _metric_ci(metric)
        value = metric.value * 100
        values.append(value)
        errors_plus.append(ci_high * 100 - value)
        errors_minus.append(value - ci_low * 100)
    raw = np.array(values)
    return StabilityData(
        metric_key, seeds, values, errors_plus, errors_minus, float(np.mean(raw)), float(np.std(raw)), color
    )


def _create_depth_series(metric: Any, name: str, color: str, mode: str) -> PlotSeries:
    by_stratum = getattr(metric, "by_stratum", getattr(metric, "by_group", {}))
    x_values: list[int | float | str] = []
    y_values: list[float] = []
    y_upper: list[float] = []
    y_lower: list[float] = []
    custom: list[list[Any]] = []
    for key in sorted(by_stratum, key=depth_group_sort_key):
        result = by_stratum[key]
        ci_low, ci_high = _metric_ci(result)
        value = result.value * 100
        x_values.append(depth_group_value(key))
        y_values.append(value)
        y_upper.append(ci_high * 100 - value)
        y_lower.append(value - ci_low * 100)
        custom.append([result.count, ci_low * 100, ci_high * 100, _reliability_code(result)])
    return PlotSeries(name, x_values, y_values, y_upper, y_lower, color, custom, mode)


def _solvability_metric(stats: Any) -> Any:
    return (
        getattr(stats, "solv_0", None)
        or getattr(stats, "stock_termination", None)
        or getattr(stats, "tier_0_validity", None)
    )


def _overall_metric(metric: Any) -> MetricSummary | None:
    if metric is None:
        return None
    return getattr(metric, "overall", metric)


def _metric_ci(metric: MetricSummary) -> tuple[float, float]:
    return (
        metric.ci_low if metric.ci_low is not None else metric.value,
        metric.ci_high if metric.ci_high is not None else metric.value,
    )


def _reliability_code(metric: MetricSummary) -> str:
    if metric.reliability is None:
        return ""
    return metric.reliability.code


__all__ = [
    "HeatmapData",
    "PlotSeries",
    "SplitHeatmapData",
    "StabilityData",
    "stats_to_comparison_series",
    "stats_to_diagnostic_series",
    "stats_to_heatmap_matrix",
    "stats_to_overall_series",
    "stats_to_stability_data",
]
