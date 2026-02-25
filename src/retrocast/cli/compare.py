"""
Handlers for the `retrocast compare` subcommand family.

YAML config schema for pareto-frontier
---------------------------------------
benchmark: mkt-cnv-160          # used to build paths in shorthand mode
stock: n5-stock                 # used to build paths in shorthand mode
top_k: 10
x_axis: cost                    # cost (default) or time
output_dir: ./6-comparisons/my-run   # relative to the yaml file

# optional — draw a solid line connecting models that share a group id
groups:
  - id: ariadne-1-preview
    color: "#7c3aed"

sources:
  # shorthand — paths built from root + benchmark + model name + stock
  - root: /path/to/data/retrocast
    models:
      - name: dms-explorer-xl
        hourly_cost: 1.29
        color: "#5eaff2"
        legend: DMS Explorer XL
        short: DMS XL

  # explicit path override — no root needed
  - models:
      - name: rc-r3-causal
        statistics: /path/to/.../statistics.json.gz
        hourly_cost: 0.50
        color: "#7c3aed"
        group: ariadne-1-preview
        legend: "Ariadne-1-Preview (beam 5)"
        short: "Ariadne-1-Preview<br>beam 5"
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from retrocast.exceptions import ConfigurationError
from retrocast.io.blob import load_json_gz
from retrocast.models.stats import ModelStatistics
from retrocast.utils.logging import logger


def _resolve_statistics_path(entry: dict, source_root: Path | None, benchmark: str, stock: str, yaml_dir: Path) -> Path:
    """Return the statistics.json.gz path for a model entry.

    Explicit ``statistics`` key wins; otherwise build from source root.
    Relative paths are resolved against the yaml file's directory.
    """
    if "statistics" in entry:
        p = Path(entry["statistics"])
        return p if p.is_absolute() else (yaml_dir / p).resolve()

    if source_root is None:
        raise ValueError(f"Model '{entry.get('name')}' must have either 'statistics' or a source-level 'root'")

    return source_root / "5-results" / benchmark / entry["name"] / stock / "statistics.json.gz"


def _load_sources(
    cfg: dict,
    yaml_dir: Path,
) -> tuple[list[ModelStatistics], dict[str, float], dict[str, dict], dict[str, str]]:
    """Load and validate all model statistics from the YAML ``sources`` block.

    Returns a 4-tuple of:
      - stats_list:    validated ModelStatistics objects (missing/corrupt entries skipped)
      - hourly_costs:  model_name -> hourly cost (only entries that declare one)
      - model_config:  model_name -> {legend, short, color}
      - model_groups:  model_name -> group_id (only entries that declare a group)
    """
    benchmark: str = cfg["benchmark"]
    stock: str = cfg["stock"]

    stats_list: list[ModelStatistics] = []
    hourly_costs: dict[str, float] = {}
    model_config: dict[str, dict] = {}
    model_groups: dict[str, str] = {}

    for source in cfg["sources"]:
        raw_root = source.get("root")
        source_root: Path | None = None
        if raw_root is not None:
            p = Path(raw_root)
            source_root = p if p.is_absolute() else (yaml_dir / p).resolve()

        for entry in source["models"]:
            name: str = entry["name"]
            stats_path = _resolve_statistics_path(entry, source_root, benchmark, stock, yaml_dir)

            if not stats_path.exists():
                logger.warning(f"[yellow]Missing statistics[/]: {name} ({stats_path})")
                continue

            try:
                stats = ModelStatistics.model_validate(load_json_gz(stats_path))
                stats_list.append(stats)
            except Exception as e:
                logger.error(f"[red]Failed to load {name}[/]: {e}")
                continue

            if "hourly_cost" in entry:
                hourly_costs[name] = entry["hourly_cost"]

            model_config[name] = {
                "legend": entry.get("legend", name),
                "short": entry.get("short", name),
                "color": entry.get("color", "#888888"),
            }

            if "group" in entry:
                model_groups[name] = entry["group"]

    return stats_list, hourly_costs, model_config, model_groups


def handle_pareto_frontier(args: argparse.Namespace) -> None:
    try:
        import plotly.graph_objects as go

        from retrocast.visualization import plots
    except ImportError as e:
        raise ImportError("pareto-frontier requires the 'viz' extra: uv sync --extra viz") from e

    config_path = Path(args.config).resolve()
    yaml_dir = config_path.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    top_k: int = cfg.get("top_k", 10)
    # CLI --time flag overrides yaml x_axis setting
    time_based: bool = args.time or (cfg.get("x_axis", "cost") == "time")
    auto_open: bool = not args.no_open

    output_dir = yaml_dir / cfg.get("output_dir", "./6-comparisons")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # group_id -> color
    group_colors: dict[str, str] = {g["id"]: g["color"] for g in cfg.get("groups", [])}
    # group_id -> list of (x_value, accuracy) — populated after plotting
    group_points: dict[str, list[tuple[float, float]]] = defaultdict(list)

    stats_list, hourly_costs, model_config, model_groups = _load_sources(cfg, yaml_dir)

    if not stats_list:
        logger.error("[bold red]No valid statistics loaded. Exiting.[/]")
        return

    logger.info(f"Loaded [green]{len(stats_list)}[/] models. Generating Pareto frontier...")

    # --- Plot ---
    fig = plots.plot_pareto_frontier(
        models_stats=stats_list,
        model_config=model_config,
        hourly_costs=hourly_costs,
        k=top_k,
        time_based=time_based,
    )

    # --- Group connecting lines ---
    # Reconstruct (x, y) for each grouped model from the figure traces so we
    # don't duplicate the x-axis calculation logic.
    if model_groups:
        # Build name -> (x, y) from the scatter traces plot_pareto_frontier added
        point_coords: dict[str, tuple[float, float]] = {}
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.x and len(trace.x) == 1:  # noqa: SIM102
                # single-point traces are the model dots; customdata[0][0] is model_name
                if trace.customdata and len(trace.customdata) > 0:
                    model_name = trace.customdata[0][0]
                    point_coords[model_name] = (trace.x[0], trace.y[0])

        for name, group_id in model_groups.items():
            if name in point_coords:
                group_points[group_id].append(point_coords[name])

        for group_id, points in group_points.items():
            if len(points) < 2:
                continue
            points.sort(key=lambda p: p[0])
            color = group_colors.get(group_id, "#888888")
            # Convert hex to rgba for opacity; only #RRGGBB format is supported
            try:
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            except ValueError as e:
                raise ConfigurationError(f"Group color must be in #RRGGBB hex format, got: {color!r}") from e
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.4)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    suffix = "_time" if time_based else ""
    html_path = output_dir / f"pareto_frontier{suffix}.html"
    pdf_path = output_dir / f"pareto_frontier{suffix}.pdf"

    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("choreographer").setLevel(logging.WARNING)

    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=auto_open)
    fig.write_image(pdf_path)
    logger.info(f"[bold green]Done![/] Saved to: [underline]{html_path}[/]")
