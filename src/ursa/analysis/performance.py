import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
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
X_AXIS, Y_AXIS = "sol_plus", "cc"
TEXT_OFFSET_DELTA_X = 0.01
TEXT_OFFSET_DELTA_Y = 0.5


# --- typed configuration models ---
class ModelDisplayConfig(BaseModel):
    legend_name: str
    abbreviation: str
    color: str


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
    mapping: dict[str, str] = {}
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
    display_map: dict[str, ModelDisplayConfig] = {}
    for model_id, model_name in model_id_to_name.items():
        if model_name in model_configs:
            display_map[model_id] = model_configs[model_name]
        else:
            display_map[model_id] = ModelDisplayConfig(
                abbreviation=model_name, legend_name=model_name, color=DEFAULT_COLOR
            )
    return display_map


def load_benchmark_data(csv_path: Path) -> pd.DataFrame:
    """loads and cleans benchmark data from a csv file."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)
    df["dataset"] = df["dataset"].str.strip().map(DATASET_MAP)
    df = df.dropna(subset=["dataset", X_AXIS, Y_AXIS])
    df[X_AXIS] = pd.to_numeric(df[X_AXIS])
    df[Y_AXIS] = pd.to_numeric(df[Y_AXIS])
    return df


def _create_trace(
    *, model_data: pd.Series, display_settings: ModelDisplayConfig, model_id: str, **kwargs: Any
) -> go.Scatter:
    """helper to create a single plotly scatter trace."""
    return go.Scatter(
        x=[model_data[X_AXIS]],  # wrap in a list
        y=[model_data[Y_AXIS]],  # wrap in a list
        mode="markers+text",
        text=[display_settings.abbreviation],
        textposition="top center",
        name=display_settings.legend_name,
        hovertemplate=f"<b>{display_settings.legend_name}</b> ({model_id})<br>{X_AXIS}: %{{x}}<br>{Y_AXIS}: %{{y}}<extra></extra>",
        marker={"color": display_settings.color},
        **kwargs,
    )


def _generate_plot_for_dataset(
    *,
    fig: go.Figure,
    df_dataset: pd.DataFrame,
    model_display_map: dict[str, ModelDisplayConfig],
    row: int | None = None,
    col: int | None = None,
    **trace_kwargs: Any,
):
    """adds traces for all models in a given dataset to a figure."""
    for model_id, model_data in df_dataset.groupby("model_id"):
        model_id_str = str(model_id)
        display_settings = model_display_map.get(
            model_id_str, ModelDisplayConfig(abbreviation=model_id_str, legend_name=model_id_str, color=DEFAULT_COLOR)
        )
        trace = _create_trace(
            model_data=model_data.iloc[0],
            display_settings=display_settings,
            model_id=model_id_str,
            legendgroup=display_settings.legend_name,
            **trace_kwargs,
        )
        fig.add_trace(trace, row=row, col=col)


def plot_performance_summary(
    *,
    df: pd.DataFrame,
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
        df_dataset = df[df["dataset"] == dataset]
        _generate_plot_for_dataset(
            fig=fig,
            df_dataset=df_dataset,
            model_display_map=model_display_map,
            row=i,
            col=1,
            showlegend=(i == 1),
        )

    fig.update_layout(height=400 * len(datasets))

    fig.update_xaxes(title_text=plot_settings.x_axis.title, row=len(datasets), col=1)
    for i in range(1, len(datasets) + 1):
        fig.update_yaxes(title_text=plot_settings.y_axis.title, range=plot_settings.y_axis.range, row=i, col=1)

    Styler(legend_size=14).apply_style(fig)
    output_path = output_dir / "performance_summary.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    logging.info(f"-> saved combined plot to {output_path}")


def plot_performance_by_dataset(
    *,
    df: pd.DataFrame,
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
        df_dataset = df[df["dataset"] == dataset]
        _generate_plot_for_dataset(fig=fig, df_dataset=df_dataset, model_display_map=model_display_map)

        x_range = [df_dataset[X_AXIS].min(), df_dataset[X_AXIS].max() + pad.x]
        y_range = [df_dataset[Y_AXIS].min(), df_dataset[Y_AXIS].max() + pad.y]

        fig.update_layout(
            title=f"Performance on {dataset}",
            xaxis_title=X_AXIS,
            yaxis_title=Y_AXIS.upper(),
            legend_title="Model",
            xaxis_range=x_range,
            yaxis_range=y_range,
        )
        Styler().apply_style(fig)
        output_path = output_dir / f"performance_{dataset}.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
        logging.info(f"-> saved separate plot to {output_path}")
