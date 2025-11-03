import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml
from ischemist.plotly import Styler
from plotly.subplots import make_subplots

# a sane mapping for the absolutely bonkers csv headers.
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

# map csv dataset shorthand to a more descriptive name
DATASET_MAP = {
    "uspto": "uspto-190",
    "bridge": "ursa-bridge-100",
    "expert": "ursa-expert-100",
}


def discover_model_names(base_path: Path) -> dict[str, str]:
    """
    scans for manifest.json files to map model hashes to model names.

    returns a dictionary mapping the hash suffix (model_id) to model_name.
    """
    mapping = {}
    if not base_path.is_dir():
        print(f"warning: data path for model discovery not found: {base_path}")
        return mapping

    for manifest_path in base_path.glob("**/ursa-model-*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            model_hash = manifest.get("model_hash")  # e.g., "ursa-model-22f978fe"
            model_name = manifest.get("model_name")  # e.g., "dms-wide-fp16"

            if model_hash and model_name:
                # the csv 'model id' is just the hash suffix
                model_id = model_hash.replace("ursa-model-", "")
                if model_id in mapping and mapping[model_id] != model_name:
                    print(f"warning: conflicting name for {model_id}: '{mapping[model_id]}' vs '{model_name}'")
                mapping[model_id] = model_name
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: failed to read manifest at {manifest_path}: {e}")

    return mapping


def load_visualization_config(config_path: Path) -> dict:
    """
    loads visualization configuration from a yaml file.

    returns a dictionary with model display settings (legend_name, abbreviation, color)
    and plot settings (shared_xaxes, shared_yaxes, etc.).
    """
    if not config_path.exists():
        print(f"warning: visualization config not found at {config_path}")
        return {"models": {}, "plot_settings": {}}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config if config else {"models": {}, "plot_settings": {}}
    except (OSError, yaml.YAMLError) as e:
        print(f"warning: failed to load visualization config at {config_path}: {e}")
        return {"models": {}, "plot_settings": {}}


def build_model_display_map(model_id_to_name: dict[str, str], viz_config: dict) -> dict:
    """
    builds a map from model_id to display settings (abbreviation, legend_name, color).

    uses model names as the key in the visualization config, then maps back to model_ids.

    args:
        model_id_to_name: mapping from model_id (csv) to model_name (from manifests)
        viz_config: visualization config dict with 'models' key containing model_name -> settings

    returns a dict mapping model_id to display settings dict with keys:
        - abbreviation: short name for plots
        - legend_name: full name for legend
        - color: hex color code
    """
    model_config = viz_config.get("models", {})
    display_map = {}

    for model_id, model_name in model_id_to_name.items():
        if model_name in model_config:
            display_map[model_id] = model_config[model_name]
        else:
            # fallback: use model name as abbreviation if not in config
            display_map[model_id] = {
                "abbreviation": model_name,
                "legend_name": model_name,
                "color": "#808080",  # default gray
            }

    return display_map


def get_ordered_datasets(datasets: list[str], viz_config: dict) -> list[str]:
    """
    orders datasets according to the configured order in visualization config.

    any datasets not in the config's dataset_order list will be appended at the end
    in alphabetical order.

    args:
        datasets: list of dataset names from the dataframe
        viz_config: visualization config dict with 'plot_settings' containing 'dataset_order'

    returns: list of datasets in the configured order
    """
    plot_settings = viz_config.get("plot_settings", {})
    config_order = plot_settings.get("dataset_order", [])

    # ensure we have all datasets
    dataset_set = set(datasets)
    ordered = []

    # add datasets in configured order
    for dataset in config_order:
        if dataset in dataset_set:
            ordered.append(dataset)
            dataset_set.remove(dataset)

    # append any remaining datasets in alphabetical order
    ordered.extend(sorted(dataset_set))

    return ordered


def load_benchmark_data(csv_path: Path) -> pd.DataFrame:
    """
    loads and cleans benchmark data from a csv file.

    returns a pandas dataframe with standardized column names.
    """
    df = pd.read_csv(csv_path)
    # clean up whitespace that will DEFINITELY break things later
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)

    # map to consistent dataset names and drop any rows that didn't match
    df["dataset"] = df["dataset"].str.strip().map(DATASET_MAP)
    df = df.dropna(subset=["dataset", "sol_plus_n", "cc"])

    # ensure numeric types
    df["sol_plus_n"] = pd.to_numeric(df["sol_plus_n"])
    df["cc"] = pd.to_numeric(df["cc"])

    return df


def _create_trace(model_data, display_settings, model_id, x_ax, y_ax, **kwargs):
    """helper to create a single plotly scatter trace.

    args:
        model_data: pandas DataFrame with x/y data for this model
        display_settings: dict with 'abbreviation', 'legend_name', 'color' keys
        model_id: the model's id (for hover text)
        x_ax, y_ax: column names for axes
        **kwargs: additional plotly Scatter kwargs
    """
    abbreviation = display_settings.get("abbreviation", model_id)
    legend_name = display_settings.get("legend_name", abbreviation)
    color = display_settings.get("color", "#808080")

    return go.Scatter(
        x=model_data[x_ax],
        y=model_data[y_ax],
        mode="markers+text",
        text=[abbreviation],
        textposition="top center",
        name=legend_name,
        hovertemplate=f"<b>{legend_name}</b> ({model_id})<br>{x_ax}: %{{x}}<br>{y_ax}: %{{y}}<extra></extra>",
        marker={"color": color},
        **kwargs,
    )


def plot_performance_summary(
    df: pd.DataFrame,
    model_display_map: dict[str, dict],
    output_dir: Path,
    shared_xaxes: bool = True,
    shared_yaxes: bool = False,
    viz_config: dict | None = None,
):
    """
    generates a single figure with one subplot per dataset.

    args:
        df: benchmark dataframe
        model_display_map: mapping from model_id to display settings dict
        output_dir: output directory
        shared_xaxes: whether to share x-axes across subplots
        shared_yaxes: whether to share y-axes across subplots
        viz_config: visualization config dict for dataset ordering (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_datasets = list(df["dataset"].unique())
    if viz_config:
        datasets = get_ordered_datasets(all_datasets, viz_config)
    else:
        datasets = sorted(all_datasets)
    x_ax, y_ax = "sol_plus_n", "cc"

    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        subplot_titles=datasets,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=0.08,
    )

    for i, dataset in enumerate(datasets, start=1):
        df_dataset = df[df["dataset"] == dataset]
        for model_id in df_dataset["model_id"].unique():
            model_data = df_dataset[df_dataset["model_id"] == model_id]
            display_settings = model_display_map.get(
                model_id, {"abbreviation": model_id, "legend_name": model_id, "color": "#808080"}
            )
            trace = _create_trace(
                model_data,
                display_settings,
                model_id,
                x_ax,
                y_ax,
                legendgroup=display_settings.get("legend_name", model_id),
                showlegend=(i == 1),  # show legend only for the first subplot
            )
            fig.add_trace(trace, row=i, col=1)

    fig.update_layout(title_text="Model Performance: Sol+(N) vs. CC", height=400 * len(datasets), legend_title="Model")
    fig.update_xaxes(title_text="Sol+(N)", row=len(datasets), col=1)
    for i in range(1, len(datasets) + 1):
        fig.update_yaxes(title_text="CC", row=i, col=1)

    Styler().apply_style(fig)
    output_path = output_dir / "performance_summary.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"-> saved combined plot to {output_path}")


def plot_performance_by_dataset(
    df: pd.DataFrame, model_display_map: dict[str, dict], output_dir: Path, viz_config: dict | None = None
):
    """
    generates a separate figure for each dataset.

    args:
        df: benchmark dataframe
        model_display_map: mapping from model_id to display settings dict
        output_dir: output directory
        viz_config: visualization config dict for dataset ordering (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_datasets = list(df["dataset"].unique())
    if viz_config:
        datasets = get_ordered_datasets(all_datasets, viz_config)
    else:
        datasets = sorted(all_datasets)
    x_ax, y_ax = "sol_plus_n", "cc"

    for dataset in datasets:
        fig = go.Figure()
        df_dataset = df[df["dataset"] == dataset]

        for model_id in df_dataset["model_id"].unique():
            model_data = df_dataset[df_dataset["model_id"] == model_id]
            display_settings = model_display_map.get(
                model_id, {"abbreviation": model_id, "legend_name": model_id, "color": "#808080"}
            )
            trace = _create_trace(model_data, display_settings, model_id, x_ax, y_ax)
            fig.add_trace(trace)

        fig.update_layout(
            title=f"Performance on {dataset}",
            xaxis_title="Sol+(N)",
            yaxis_title="CC",
            legend_title="Model",
        )
        Styler().apply_style(fig)
        output_path = output_dir / f"performance_{dataset}.html"
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"-> saved separate plot to {output_path}")
