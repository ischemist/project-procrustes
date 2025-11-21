"""
Visualization theme and style definitions.

This module centralizes all color choices and layout configurations.
It acts as a local configuration layer on top of `ischemist`.
"""

import hashlib
from typing import Any

import plotly.graph_objects as go
from ischemist.colors import ColorPalette
from ischemist.plotly import Styler

# --- Configuration: Colors ---

# A temporary palette for models. Tweak these hex codes directly.
# Once satisfied, move them to ischemist.
_MODEL_COLORS_HEX = [
    "#1f77b4",  # Muted Blue
    "#ff7f0e",  # Safety Orange
    "#2ca02c",  # Cooked Asparagus Green
    "#d62728",  # Brick Red
    "#9467bd",  # Muted Purple
    "#8c564b",  # Chestnut Brown
    "#e377c2",  # Raspberry Yogurt Pink
    "#7f7f7f",  # Middle Gray
    "#bcbd22",  # Curry Yellow-Green
    "#17becf",  # Blue-Teal
]

MODEL_PALETTE = ColorPalette.from_hex_codes(_MODEL_COLORS_HEX)

# Specific Metric Colors
COLOR_SOLVABILITY = "#b892ff"  # Medium Slate Blue
COLOR_TOP_1 = "#ffc2e2"  # Pastel Red
COLOR_TOP_5 = "#ff90b3"  # Medium Turquoise
COLOR_TOP_10 = "#ef7a85"  # Dark Teal
COLOR_DEFAULT = "#95a5a6"  # Concrete Gray

# --- Logic: Color Assignment ---


def get_model_color(model_name: str) -> str:
    """
    Returns a deterministic color for a given model name.
    Uses hashing to ensure the same model always gets the same color
    across different plots, regardless of execution order.
    """
    # Use MD5 to get a consistent integer from the string
    hash_obj = hashlib.md5(model_name.encode("utf-8"))
    hash_int = int(hash_obj.hexdigest(), 16)

    # Cycle through the palette
    index = hash_int % len(MODEL_PALETTE)
    return MODEL_PALETTE[index].hex_code


def get_metric_color(metric_name: str, k: int | None = None) -> str:
    """Returns the standard color for a specific metric."""
    clean_name = metric_name.lower().strip()

    if "solvability" in clean_name:
        return COLOR_SOLVABILITY

    if k is not None or "top" in clean_name:
        # Try to parse K if not provided
        if k is None:
            try:
                # Assumes format "top-5" or "top 5"
                k = int(clean_name.replace("top", "").replace("-", "").strip())
            except ValueError:
                k = 0

        if k == 1:
            return COLOR_TOP_1
        if k == 5:
            return COLOR_TOP_5
        if k == 10:
            return COLOR_TOP_10

    return COLOR_DEFAULT


# --- Logic: Layout ---


def apply_layout(
    fig: go.Figure,
    height: int = 600,
    title: str | None = None,
    width: int | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    legend_top: bool = True,
) -> go.Figure:
    """
    Applies the standard RetroCast layout configuration.
    Wraps Styler().apply_style() with project-specific defaults.
    """
    layout_args: dict[str, Any] = dict(
        height=height,
        margin=dict(t=30),
    )

    if title:
        layout_args["title"] = title
        layout_args["margin"]["t"] = 50

    if x_title:
        layout_args.setdefault("xaxis", {})["title"] = x_title

    if y_title:
        layout_args.setdefault("yaxis", {})["title"] = y_title

    if width:
        layout_args["width"] = width

    if legend_top:
        layout_args["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

    fig.update_layout(**layout_args)

    # Apply the base ischemist style (backgrounds, fonts, etc.)
    Styler().apply_style(fig)

    return fig
