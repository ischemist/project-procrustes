import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from ischemist.plotly import Styler
from plotly.subplots import make_subplots

# a sane mapping for the absolutely bonkers csv headers.
# we can move this to a config later if it becomes a whole thing.
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
    "bridge": "ursa-bridge-100",
    "expert": "ursa-expert-100",
    "uspto": "uspto-190",
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


def _create_trace(model_data, abbreviation, model_id, x_ax, y_ax, **kwargs):
    """helper to create a single plotly scatter trace."""
    return go.Scatter(
        x=model_data[x_ax],
        y=model_data[y_ax],
        mode="markers+text",
        text=[abbreviation],
        textposition="top center",
        name=abbreviation,
        hovertemplate=f"<b>{abbreviation}</b> ({model_id})<br>{x_ax}: %{{x}}<br>{y_ax}: %{{y}}<extra></extra>",
        **kwargs,
    )


def plot_performance_summary(df: pd.DataFrame, model_abbreviations: dict[str, str], output_dir: Path):
    """
    generates a single figure with one subplot per dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(df["dataset"].unique())
    x_ax, y_ax = "sol_plus_n", "cc"

    fig = make_subplots(rows=len(datasets), cols=1, subplot_titles=datasets, shared_xaxes=True, vertical_spacing=0.08)

    for i, dataset in enumerate(datasets, start=1):
        df_dataset = df[df["dataset"] == dataset]
        for model_id in df_dataset["model_id"].unique():
            model_data = df_dataset[df_dataset["model_id"] == model_id]
            abbreviation = model_abbreviations.get(model_id, model_id)
            trace = _create_trace(
                model_data,
                abbreviation,
                model_id,
                x_ax,
                y_ax,
                legendgroup=abbreviation,
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


def plot_performance_by_dataset(df: pd.DataFrame, model_abbreviations: dict[str, str], output_dir: Path):
    """
    generates a separate figure for each dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(df["dataset"].unique())
    x_ax, y_ax = "sol_plus_n", "cc"

    for dataset in datasets:
        fig = go.Figure()
        df_dataset = df[df["dataset"] == dataset]

        for model_id in df_dataset["model_id"].unique():
            model_data = df_dataset[df_dataset["model_id"] == model_id]
            abbreviation = model_abbreviations.get(model_id, model_id)
            trace = _create_trace(model_data, abbreviation, model_id, x_ax, y_ax)
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
