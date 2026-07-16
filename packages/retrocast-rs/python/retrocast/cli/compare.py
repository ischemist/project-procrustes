from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from retrocast.exceptions import ConfigurationError, InputError
from retrocast.io import load_analysis_report
from retrocast.models.analysis import MetricSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParetoPoint:
    model_name: str
    label: str
    short_label: str
    color: str
    x_value: float
    metric: MetricSummary
    group: str | None = None


def handle_pareto_frontier(args: argparse.Namespace) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise InputError(
            "compare pareto-frontier requires the 'viz' extra: install retrocast[viz]",
            code="input.missing_optional_dependency",
        ) from exc

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    metric_name = str(config.get("metric") or _default_metric_name(config))
    time_based = bool(args.time or config.get("x_axis") == "time")
    points = _load_points(config, config_path.parent, metric_name=metric_name, time_based=time_based)
    if not points:
        raise InputError("no analysis reports with the requested metric were loaded", code="input.empty_comparison")

    fig = go.Figure()
    for point in points:
        fig.add_trace(
            go.Scatter(
                x=[point.x_value],
                y=[point.metric.value],
                mode="markers+text",
                name=point.label,
                text=[point.short_label],
                textposition="top center",
                marker={"color": point.color, "size": 11},
                customdata=[[point.model_name, point.metric.count, point.metric.ci_low, point.metric.ci_high]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    f"{_x_axis_title(time_based)}: %{{x}}<br>"
                    f"{metric_name}: %{{y:.3f}}<br>"
                    "n: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    _add_group_lines(fig, points, config)
    fig.update_layout(
        title=str(config.get("title") or f"pareto frontier: {metric_name}"),
        xaxis_title=_x_axis_title(time_based),
        yaxis_title=metric_name,
        template="plotly_white",
    )

    output_dir = _resolve_output_dir(config, config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-time" if time_based else "-cost"
    output_stem = str(config.get("output_file") or "pareto_frontier")
    html_path = output_dir / f"{output_stem}{suffix}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=not args.no_open)
    logger.info("saved comparison plot to %s", html_path)
    return html_path


def _load_config(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"failed to parse compare config {path}",
            code="config.parse_error",
            context={"config_path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(
            "compare config must be a mapping",
            code="config.invalid_shape",
            context={"config_path": str(path)},
        )
    return payload


def _default_metric_name(config: dict[str, Any]) -> str:
    stock = config.get("stock")
    top_k = config.get("top_k")
    if stock is not None and top_k is not None:
        return f"acceptable_reconstruction_top_{int(top_k)}[{stock}]"
    if stock is not None:
        return f"solv_0[{stock}]_rate"
    raise ConfigurationError(
        "compare config must define metric, or define stock with optional top_k",
        code="config.missing_metric",
    )


def _load_points(
    config: dict[str, Any],
    config_dir: Path,
    *,
    metric_name: str,
    time_based: bool,
) -> list[ParetoPoint]:
    benchmark = str(config.get("benchmark") or "")
    stock = str(config.get("stock") or "")
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ConfigurationError("compare config requires a sources list", code="config.missing_sources")

    points = []
    for source in sources:
        if not isinstance(source, dict):
            raise ConfigurationError("compare sources must be mappings", code="config.invalid_source")
        source_root = _optional_path(source.get("root"), config_dir)
        models = source.get("models")
        if not isinstance(models, list):
            raise ConfigurationError("compare source requires a models list", code="config.missing_models")
        for model in models:
            if not isinstance(model, dict):
                raise ConfigurationError("compare models must be mappings", code="config.invalid_model")
            point = _load_point(
                model,
                source_root=source_root,
                benchmark=benchmark,
                stock=stock,
                config_dir=config_dir,
                metric_name=metric_name,
                time_based=time_based,
            )
            if point is not None:
                points.append(point)
    return sorted(points, key=lambda point: point.x_value)


def _load_point(
    model: dict[str, Any],
    *,
    source_root: Path | None,
    benchmark: str,
    stock: str,
    config_dir: Path,
    metric_name: str,
    time_based: bool,
) -> ParetoPoint | None:
    model_name = str(model.get("name") or "")
    if not model_name:
        raise ConfigurationError("compare model requires name", code="config.missing_model_name")

    analysis_path = _resolve_analysis_path(model, source_root, benchmark, stock, config_dir)
    if not analysis_path.exists():
        logger.warning("missing analysis report for %s: %s", model_name, analysis_path)
        return None

    report = load_analysis_report(analysis_path)
    metric = report.metrics.get(metric_name)
    if metric is None:
        logger.warning("missing metric %s for %s", metric_name, model_name)
        return None

    return ParetoPoint(
        model_name=model_name,
        label=str(model.get("legend") or model_name),
        short_label=str(model.get("short") or model_name),
        color=str(model.get("color") or "#666666"),
        x_value=_model_x_value(model, time_based=time_based),
        metric=metric,
        group=str(model["group"]) if "group" in model else None,
    )


def _resolve_analysis_path(
    model: dict[str, Any],
    source_root: Path | None,
    benchmark: str,
    stock: str,
    config_dir: Path,
) -> Path:
    explicit_path = model.get("analysis")
    if explicit_path is None:
        explicit_path = model.get("statistics")
    if explicit_path is not None:
        path = Path(str(explicit_path))
        return path if path.is_absolute() else (config_dir / path).resolve()
    if source_root is None:
        raise ConfigurationError(
            f"model '{model.get('name')}' must set analysis or use a source root",
            code="config.missing_analysis_path",
        )
    return source_root / "5-results" / benchmark / str(model["name"]) / stock / "analysis.json.gz"


def _model_x_value(model: dict[str, Any], *, time_based: bool) -> float:
    key = "wall_time_seconds" if time_based else "hourly_cost"
    value = model.get(key)
    if value is None:
        raise ConfigurationError(f"compare model requires {key}", code="config.missing_x_value")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"compare model {key} must be numeric", code="config.invalid_x_value") from exc


def _optional_path(value: object, config_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (config_dir / path).resolve()


def _resolve_output_dir(config: dict[str, Any], config_dir: Path) -> Path:
    output_dir = Path(str(config.get("output_dir") or "6-comparisons"))
    return output_dir if output_dir.is_absolute() else (config_dir / output_dir).resolve()


def _x_axis_title(time_based: bool) -> str:
    return "wall time seconds" if time_based else "hourly cost"


def _add_group_lines(fig: Any, points: list[ParetoPoint], config: dict[str, Any]) -> None:
    group_colors = {
        str(group["id"]): str(group["color"])
        for group in config.get("groups", [])
        if isinstance(group, dict) and "id" in group and "color" in group
    }
    grouped: dict[str, list[ParetoPoint]] = {}
    for point in points:
        if point.group is not None:
            grouped.setdefault(point.group, []).append(point)
    for group_id, group_points in grouped.items():
        if len(group_points) < 2:
            continue
        ordered = sorted(group_points, key=lambda point: point.x_value)
        fig.add_trace(
            {
                "type": "scatter",
                "x": [point.x_value for point in ordered],
                "y": [point.metric.value for point in ordered],
                "mode": "lines",
                "line": {"color": group_colors.get(group_id, "#999999"), "width": 2},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
