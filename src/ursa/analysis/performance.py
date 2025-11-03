import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import yaml
from ischemist.plotly import Styler
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

# --- constants ---
COLUMN_MAP = {
    "Model Id": "model_id",
    "dataset": "dataset",
    "Sol(N)": "sol_n",
    "Sol+(N)-noFPcheck": "sol_plus_n_nofp",
    "Sol+ no FPcheck": "sol_plus_nofp",
    "CC noFPcheck": "cc_nofp",
    "Sol+(N)": "sol_plus_n",
    "Sol+": "sol_plus",
    "CC": "cc",
    "comment": "comment",
}
DATASET_MAP = {"uspto": "uspto-190", "bridge": "ursa-bridge-100", "expert": "ursa-expert-100"}
DEFAULT_COLOR = "#808080"
TEXT_OFFSET_DELTA_X = 0.01
TEXT_OFFSET_DELTA_Y = 0.5


# --- typed configuration models ---
class ModelDisplayConfig(BaseModel):
    legend_name: str
    abbreviation: str
    color: str
    text_position: str = "top center"


class CombinedFigureConfig(BaseModel):
    shared_xaxes: bool = True
    shared_yaxes: bool = True


class RangePaddingConfig(BaseModel):
    x: float = 0.0
    y: float = 0.0


class AxisConfig(BaseModel):
    title: str = ""
    tickformat: str = ""
    range: list[float] = Field(default_factory=list)


class PlotSettingsConfig(BaseModel):
    input_data_path: str
    processed_data_path: str
    output_dir: str
    combined_figure: CombinedFigureConfig = Field(default_factory=CombinedFigureConfig)
    dataset_order: list[str] = Field(default_factory=list)
    range_padding: RangePaddingConfig = Field(default_factory=RangePaddingConfig)

    x_axis: AxisConfig = Field(default_factory=AxisConfig)
    y_axis: AxisConfig = Field(default_factory=AxisConfig)


class VisualizationConfig(BaseModel):
    models: dict[str, ModelDisplayConfig]
    plot_settings: PlotSettingsConfig


def load_visualization_config(config_path: Path) -> VisualizationConfig:
    """loads and validates visualization configuration from a yaml file."""
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return VisualizationConfig.model_validate(config_data)
    except (OSError, yaml.YAMLError, ValueError) as e:
        logging.error(f"failed to load or validate config at {config_path}: {e}")
        raise


def discover_model_names(base_path: Path) -> dict[str, str]:
    """scans for manifest.json files to map model hashes to model names."""
    mapping: dict[str, str] = {
        "Insilico": "Insilico",  # special case: Insilico was run internally
    }
    if not base_path.is_dir():
        logging.warning(f"data path for model discovery not found: {base_path}")
        return mapping

    for manifest_path in base_path.glob("**/ursa-model-*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            model_hash = manifest.get("model_hash")
            model_name = manifest.get("model_name")
            if model_hash and model_name:
                model_id = model_hash.replace("ursa-model-", "")
                if model_id in mapping and mapping[model_id] != model_name:
                    logging.warning(f"conflicting name for {model_id}: '{mapping[model_id]}' vs '{model_name}'")
                mapping[model_id] = model_name
        except (OSError, json.JSONDecodeError) as e:
            logging.warning(f"failed to read manifest at {manifest_path}: {e}")
    return mapping


def build_model_display_map(
    model_id_to_name: dict[str, str], model_configs: dict[str, ModelDisplayConfig]
) -> dict[str, ModelDisplayConfig]:
    """builds a map from model_id to its display settings."""
    name_to_id = {name: id for id, name in model_id_to_name.items()}
    display_map = {name_to_id[name]: model_configs[name] for name in model_configs}
    return display_map


@dataclass
class BenchmarkRecord:
    """represents a single data point for a model on a dataset."""

    model_id: str
    dataset: str
    sol_plus_n: float
    cc: float


def load_benchmark_data(csv_path: Path, x_metric: str, y_metric: str) -> list[BenchmarkRecord]:
    """loads and cleans benchmark data from a csv file into a list of dataclasses."""
    records: list[BenchmarkRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        # clean headers before passing to dictreader
        header = [h.strip() for h in f.readline().strip().split(",")]
        renamed_header = [COLUMN_MAP.get(h, h) for h in header]

        reader = csv.DictReader(f, fieldnames=renamed_header)
        for row in reader:
            dataset_raw = row.get("dataset", "").strip()
            dataset = DATASET_MAP.get(dataset_raw)
            if not dataset:
                continue

            try:
                records.append(
                    BenchmarkRecord(
                        model_id=row["model_id"].strip(),
                        dataset=dataset,
                        sol_plus_n=float(row[x_metric]),
                        cc=float(row[y_metric]),
                    )
                )
            except (ValueError, TypeError, KeyError):
                # skip rows with missing or non-numeric data
                continue
    return records


def _create_trace(*, record: BenchmarkRecord, display_settings: ModelDisplayConfig, **kwargs: Any) -> go.Scatter:
    """helper to create a single plotly scatter trace from a BenchmarkRecord."""
    return go.Scatter(
        x=[record.sol_plus_n],
        y=[record.cc],
        mode="markers+text",
        text=[display_settings.abbreviation],
        textposition=display_settings.text_position,
        textfont={"size": 12},
        name=f"({display_settings.abbreviation}) {display_settings.legend_name}",
        hovertemplate=f"<b>{display_settings.legend_name}</b> ({record.model_id})<br>Sol+: %{{x}}<br>CC: %{{y}}<extra></extra>",
        marker={"color": display_settings.color, "size": 10},
        **kwargs,
    )


def _generate_plot_for_dataset(
    *,
    fig: go.Figure,
    records: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    row: int | None = None,
    col: int | None = None,
    **trace_kwargs: Any,
):
    """adds traces for all models in a given dataset to a figure."""
    record_map = {r.model_id: r for r in records}
    for model_id, display_settings in model_display_map.items():
        if model_id in record_map:
            record = record_map[model_id]
            trace = _create_trace(
                record=record,
                display_settings=display_settings,
                legendgroup=display_settings.legend_name,
                **trace_kwargs,
            )
            fig.add_trace(trace, row=row, col=col)


def plot_performance_summary(
    *,
    records: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    plot_settings: PlotSettingsConfig,
    output_dir: Path,
):
    """generates a single figure with one subplot per dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = plot_settings.dataset_order

    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        subplot_titles=datasets,
        shared_xaxes=plot_settings.combined_figure.shared_xaxes,
        shared_yaxes=plot_settings.combined_figure.shared_yaxes,
        vertical_spacing=0.04,
    )

    for i, dataset in enumerate(datasets, start=1):
        records_for_dataset = [r for r in records if r.dataset == dataset]
        _generate_plot_for_dataset(
            fig=fig,
            records=records_for_dataset,
            model_display_map=model_display_map,
            row=i,
            col=1,
            showlegend=(i == 1),
        )

    fig.update_layout(height=400 * len(datasets))

    Styler(legend_size=14).apply_style(fig)
    fig.update_xaxes(title_text=plot_settings.x_axis.title, range=plot_settings.x_axis.range, row=len(datasets), col=1)
    for i in range(1, len(datasets) + 1):
        fig.update_yaxes(title_text=plot_settings.y_axis.title, range=plot_settings.y_axis.range, row=i, col=1)

    output_path = output_dir / "performance_summary.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    logging.info(f"-> saved combined plot to {output_path}")


def plot_performance_by_dataset(
    *,
    records: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    plot_settings: PlotSettingsConfig,
    output_dir: Path,
):
    """generates a separate figure for each dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = plot_settings.dataset_order
    pad = plot_settings.range_padding

    for dataset in datasets:
        fig = go.Figure()
        records_for_dataset = [r for r in records if r.dataset == dataset]
        _generate_plot_for_dataset(fig=fig, records=records_for_dataset, model_display_map=model_display_map)

        if not records_for_dataset:
            logging.warning(f"no records found for dataset '{dataset}', skipping plot generation.")
            continue

        x_coords = [r.sol_plus_n for r in records_for_dataset]
        y_coords = [r.cc for r in records_for_dataset]
        x_range = [min(x_coords), max(x_coords) + pad.x]
        y_range = [min(y_coords), max(y_coords) + pad.y]

        fig.update_layout(
            title=f"Performance on {dataset}",
            legend_title="Model",
            xaxis_range=x_range,
            yaxis_range=y_range,
            height=600,
        )
        Styler().apply_style(fig)
        output_path = output_dir / f"performance_{dataset}.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
        logging.info(f"-> saved separate plot to {output_path}")
