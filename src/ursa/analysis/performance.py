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
    dataset_totals: dict[str, int] = Field(default_factory=dict)
    range_padding: RangePaddingConfig = Field(default_factory=RangePaddingConfig)

    x_axis: AxisConfig = Field(default_factory=AxisConfig)
    y_axis: AxisConfig = Field(default_factory=AxisConfig)


class BarTraceConfig(BaseModel):
    name: str
    color: str


class BarSettingsConfig(BaseModel):
    title_template: str = "Prediction Funnel on {dataset}"
    sort_by_dataset: str | None = None
    x_axis_title: str = "Model"
    y_axis_title: str = "Count"
    traces: dict[str, BarTraceConfig]


class VisualizationConfig(BaseModel):
    models: dict[str, ModelDisplayConfig]
    plot_settings: PlotSettingsConfig
    bar_settings: BarSettingsConfig


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
    sol_plus: float
    cc: float
    sol_n: int
    sol_plus_n_nofp: int
    sol_plus_n: int


def load_benchmark_data(csv_path: Path) -> list[BenchmarkRecord]:
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

            records.append(
                BenchmarkRecord(
                    model_id=row["model_id"].strip(),
                    dataset=dataset,
                    sol_plus=float(row["sol_plus"]),
                    cc=float(row["cc"]),
                    sol_n=int(row["sol_n"]),
                    sol_plus_n_nofp=int(row["sol_plus_n_nofp"]),
                    sol_plus_n=int(row["sol_plus_n"]),
                )
            )
    return records


def _create_trace(*, record: BenchmarkRecord, display_settings: ModelDisplayConfig, **kwargs: Any) -> go.Scatter:
    """helper to create a single plotly scatter trace from a BenchmarkRecord."""
    return go.Scatter(
        x=[record.sol_plus],
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
    datasets = list(plot_settings.dataset_totals.keys())

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
    fig.update_layout(legend=dict(orientation="h", entrywidth=200))
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
    datasets = plot_settings.dataset_totals.keys()
    pad = plot_settings.range_padding

    for dataset in datasets:
        fig = go.Figure()
        records_for_dataset = [r for r in records if r.dataset == dataset]
        _generate_plot_for_dataset(fig=fig, records=records_for_dataset, model_display_map=model_display_map)

        if not records_for_dataset:
            logging.warning(f"no records found for dataset '{dataset}', skipping plot generation.")
            continue

        x_coords = [r.sol_plus for r in records_for_dataset]
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


def _generate_funnel_bars_for_dataset(
    *,
    fig: go.Figure,
    records_for_dataset: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    bar_settings: BarSettingsConfig,
    n_targets: int,
    canonical_model_order: list[str],
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
):
    """helper to add stacked bar traces for one dataset to a figure/subplot."""
    record_map = {r.model_id: r for r in records_for_dataset}

    model_abbrevs = [
        model_display_map.get(
            mid, ModelDisplayConfig(legend_name=mid, abbreviation=mid[:5], color=DEFAULT_COLOR)
        ).abbreviation
        for mid in canonical_model_order
    ]

    base_values = [record_map[mid].sol_plus_n / n_targets for mid in canonical_model_order]
    middle_values = [
        (record_map[mid].sol_plus_n_nofp - record_map[mid].sol_plus_n) / n_targets for mid in canonical_model_order
    ]
    top_values = [
        (record_map[mid].sol_n - record_map[mid].sol_plus_n_nofp) / n_targets for mid in canonical_model_order
    ]
    traces_config = bar_settings.traces

    fig.add_trace(
        go.Bar(
            x=model_abbrevs,
            y=base_values,
            name=traces_config["sol_plus_n"].name,
            marker_color=traces_config["sol_plus_n"].color,
            legendgroup="funnel",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            x=model_abbrevs,
            y=middle_values,
            name=traces_config["failed_fp"].name,
            marker_color=traces_config["failed_fp"].color,
            legendgroup="funnel",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            x=model_abbrevs,
            y=top_values,
            name=traces_config["failed_sanity"].name,
            marker_color=traces_config["failed_sanity"].color,
            legendgroup="funnel",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def plot_prediction_funnel(
    *,
    records: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    bar_settings: BarSettingsConfig,
    dataset_totals: dict[str, int],
    output_dir: Path,
):
    """generates a separate stacked bar chart for each dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = dataset_totals.keys() or sorted({r.dataset for r in records})
    canonical_model_order: list[str]
    sort_dataset = bar_settings.sort_by_dataset
    if sort_dataset and sort_dataset in datasets:
        logging.info(f"determining canonical model order from dataset: {sort_dataset}")
        sort_records = [r for r in records if r.dataset == sort_dataset]
        sort_records.sort(key=lambda r: r.sol_plus_n, reverse=True)
        canonical_model_order = [r.model_id for r in sort_records]
    else:
        logging.warning("sort_by_dataset not specified or not found. using alphabetical model order.")
        canonical_model_order = sorted(model_display_map.keys())

    for dataset in datasets:
        records_for_dataset = [r for r in records if r.dataset == dataset]
        if not records_for_dataset:
            logging.warning(f"no records for dataset '{dataset}', skipping funnel plot.")
            continue

        fig = go.Figure()
        _generate_funnel_bars_for_dataset(
            fig=fig,
            records_for_dataset=records_for_dataset,
            model_display_map=model_display_map,
            bar_settings=bar_settings,
            canonical_model_order=canonical_model_order,
            n_targets=dataset_totals[dataset],
        )

        fig.update_layout(
            barmode="stack",
            title=bar_settings.title_template.format(dataset=dataset),
            xaxis_title=bar_settings.x_axis_title,
            yaxis_title=bar_settings.y_axis_title,
            height=600,
        )
        Styler().apply_style(fig)

        output_path = output_dir / f"funnel_{dataset}.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
        logging.info(f"-> saved funnel plot to {output_path}")


def plot_prediction_funnel_summary(
    *,
    records: list[BenchmarkRecord],
    model_display_map: dict[str, ModelDisplayConfig],
    bar_settings: BarSettingsConfig,
    dataset_totals: dict[str, int],
    output_dir: Path,
):
    """generates a single figure with one funnel chart subplot per dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(dataset_totals.keys()) or sorted({r.dataset for r in records})
    canonical_model_order: list[str]
    sort_dataset = bar_settings.sort_by_dataset
    if sort_dataset and sort_dataset in datasets:
        logging.info(f"determining canonical model order from dataset: {sort_dataset}")
        sort_records = [r for r in records if r.dataset == sort_dataset]
        sort_records.sort(key=lambda r: r.sol_plus_n, reverse=True)
        canonical_model_order = [r.model_id for r in sort_records]
    else:
        # fallback: just use all known model ids, sorted alphabetically
        logging.warning("sort_by_dataset not specified or not found. using alphabetical model order.")
        canonical_model_order = sorted(model_display_map.keys())
    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        subplot_titles=datasets,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for i, dataset in enumerate(datasets, start=1):
        records_for_dataset = [r for r in records if r.dataset == dataset]
        _generate_funnel_bars_for_dataset(
            fig=fig,
            records_for_dataset=records_for_dataset,
            model_display_map=model_display_map,
            bar_settings=bar_settings,
            canonical_model_order=canonical_model_order,
            n_targets=dataset_totals[dataset],
            row=i,
            col=1,
            showlegend=(i == 1),  # only show legend for the first subplot
        )
        # set y-axis title for each subplot
        fig.update_yaxes(title_text=bar_settings.y_axis_title, row=i, col=1)

    # set shared layout properties
    fig.update_layout(
        barmode="stack",
        height=350 * len(datasets),
    )
    # set x-axis title only on the bottom-most plot
    fig.update_xaxes(title_text=bar_settings.x_axis_title, row=len(datasets), col=1)

    Styler().apply_style(fig)

    output_path = output_dir / "funnel_summary.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    logging.info(f"-> saved combined funnel plot to {output_path}")
