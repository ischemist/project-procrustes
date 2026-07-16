from __future__ import annotations

import hashlib
from typing import Any

from ischemist.colors import ColorPalette
from ischemist.plotly import Styler

COLOR_SOLVABILITY = "#b892ff"
COLOR_TOP_1 = "#ffc2e2"
COLOR_TOP_5 = "#ff90b3"
COLOR_TOP_10 = "#ef7a85"
COLOR_DEFAULT = "#95a5a6"

_MODEL_COLORS_HEX = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

MODEL_PALETTE = ColorPalette.from_hex_codes(_MODEL_COLORS_HEX)


def get_model_color(model_name: str) -> str:
    digest = hashlib.md5(model_name.encode("utf-8")).hexdigest()
    return MODEL_PALETTE[int(digest, 16) % len(MODEL_PALETTE)].hex_code


def get_metric_color(metric_name: str, k: int | None = None) -> str:
    clean_name = metric_name.lower().strip()
    if "solv" in clean_name:
        return COLOR_SOLVABILITY
    if k is not None or "top" in clean_name:
        if k is None:
            try:
                k = int(clean_name.replace("top", "").replace("-", "").strip())
            except ValueError:
                return COLOR_DEFAULT
        if k == 1:
            return COLOR_TOP_1
        if k == 5:
            return COLOR_TOP_5
        if k == 10:
            return COLOR_TOP_10
    return COLOR_DEFAULT


def apply_layout(
    fig: Any,
    height: int = 600,
    title: str | None = None,
    width: int | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    legend_top: bool = True,
) -> Any:
    layout_args: dict[str, Any] = {
        "height": height,
        "margin": {"t": 30},
    }
    if title:
        layout_args["title"] = title
        layout_args["margin"]["t"] = 50
    if width:
        layout_args["width"] = width
    if x_title:
        layout_args.setdefault("xaxis", {})["title"] = x_title
    if y_title:
        layout_args.setdefault("yaxis", {})["title"] = y_title
    if legend_top:
        layout_args["legend"] = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    fig.update_layout(**layout_args)
    Styler().apply_style(fig)
    return fig


__all__ = [
    "COLOR_DEFAULT",
    "COLOR_SOLVABILITY",
    "COLOR_TOP_1",
    "COLOR_TOP_10",
    "COLOR_TOP_5",
    "MODEL_PALETTE",
    "apply_layout",
    "get_metric_color",
    "get_model_color",
]
