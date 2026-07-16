from __future__ import annotations

from pathlib import Path

import pytest

from retrocast.cli.compare import (
    ParetoPoint,
    _add_group_lines,
    _default_metric_name,
    _load_config,
    _load_points,
    _model_x_value,
    _resolve_analysis_path,
    _resolve_output_dir,
    _x_axis_title,
)
from retrocast.exceptions import ConfigurationError
from retrocast.io import save_analysis_report
from retrocast.models.analysis import AnalysisReport, MetricSummary


@pytest.mark.contract
def test_compare_config_defaults_and_validation_errors(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.yaml"
    sequence = tmp_path / "sequence.yaml"
    malformed.write_text("metric: [", encoding="utf-8")
    sequence.write_text("- not-a-mapping\n", encoding="utf-8")

    assert _default_metric_name({"stock": "n5"}) == "solv_0[n5]_rate"
    assert _default_metric_name({"stock": "n5", "top_k": "3"}) == "acceptable_reconstruction_top_3[n5]"
    with pytest.raises(ConfigurationError, match="must define metric"):
        _default_metric_name({})
    with pytest.raises(ConfigurationError) as parse_error:
        _load_config(malformed)
    assert parse_error.value.code == "config.parse_error"
    with pytest.raises(ConfigurationError) as shape_error:
        _load_config(sequence)
    assert shape_error.value.code == "config.invalid_shape"


@pytest.mark.contract
def test_compare_load_points_validates_source_and_model_shapes(tmp_path: Path) -> None:
    report_path = tmp_path / "analysis.json.gz"
    save_analysis_report(AnalysisReport(metrics={"metric": MetricSummary(value=0.5, count=2)}), report_path)

    config = {
        "sources": [
            {
                "models": [
                    {
                        "name": "model-a",
                        "analysis": "analysis.json.gz",
                        "hourly_cost": "1.25",
                        "legend": "Model A",
                        "short": "A",
                        "color": "red",
                        "group": "family",
                    }
                ]
            }
        ]
    }
    points = _load_points(config, tmp_path, metric_name="metric", time_based=False)

    assert points[0].model_name == "model-a"
    assert points[0].x_value == 1.25
    assert points[0].group == "family"
    assert _x_axis_title(time_based=True) == "wall time seconds"
    assert _x_axis_title(time_based=False) == "hourly cost"
    assert _resolve_output_dir({"output_dir": "out"}, tmp_path) == tmp_path / "out"

    with pytest.raises(ConfigurationError, match="sources list"):
        _load_points({"sources": {}}, tmp_path, metric_name="metric", time_based=False)
    with pytest.raises(ConfigurationError, match="sources must be mappings"):
        _load_points({"sources": ["bad"]}, tmp_path, metric_name="metric", time_based=False)
    with pytest.raises(ConfigurationError, match="models list"):
        _load_points({"sources": [{}]}, tmp_path, metric_name="metric", time_based=False)
    with pytest.raises(ConfigurationError, match="models must be mappings"):
        _load_points({"sources": [{"models": ["bad"]}]}, tmp_path, metric_name="metric", time_based=False)
    with pytest.raises(ConfigurationError, match="requires name"):
        _load_points({"sources": [{"models": [{}]}]}, tmp_path, metric_name="metric", time_based=False)


@pytest.mark.contract
def test_compare_resolves_paths_and_x_values_with_stable_errors(tmp_path: Path) -> None:
    explicit = _resolve_analysis_path(
        {"name": "model-a", "statistics": "stats.json.gz"}, None, "bench", "stock", tmp_path
    )
    rooted = _resolve_analysis_path({"name": "model-a"}, tmp_path / "data", "bench", "stock", tmp_path)

    assert explicit == tmp_path / "stats.json.gz"
    assert rooted == tmp_path / "data" / "5-results" / "bench" / "model-a" / "stock" / "analysis.json.gz"
    assert _model_x_value({"wall_time_seconds": "12.5"}, time_based=True) == 12.5
    with pytest.raises(ConfigurationError) as missing_path:
        _resolve_analysis_path({"name": "model-a"}, None, "bench", "stock", tmp_path)
    assert missing_path.value.code == "config.missing_analysis_path"
    with pytest.raises(ConfigurationError) as missing_x:
        _model_x_value({}, time_based=False)
    assert missing_x.value.code == "config.missing_x_value"
    with pytest.raises(ConfigurationError) as invalid_x:
        _model_x_value({"hourly_cost": "not-number"}, time_based=False)
    assert invalid_x.value.code == "config.invalid_x_value"


@pytest.mark.contract
def test_compare_group_lines_connect_points_in_same_group() -> None:
    class Figure:
        def __init__(self) -> None:
            self.traces: list[dict] = []

        def add_trace(self, trace: dict) -> None:
            self.traces.append(trace)

    fig = Figure()
    points = [
        ParetoPoint("a", "A", "A", "red", 2.0, MetricSummary(value=0.2, count=1), group="family"),
        ParetoPoint("b", "B", "B", "red", 1.0, MetricSummary(value=0.4, count=1), group="family"),
        ParetoPoint("c", "C", "C", "blue", 3.0, MetricSummary(value=0.6, count=1), group="single"),
    ]

    _add_group_lines(fig, points, {"groups": [{"id": "family", "color": "purple"}]})

    assert fig.traces == [
        {
            "type": "scatter",
            "x": [1.0, 2.0],
            "y": [0.4, 0.2],
            "mode": "lines",
            "line": {"color": "purple", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
        }
    ]
