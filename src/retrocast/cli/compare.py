"""
Handlers for the `retrocast compare` subcommand family.

YAML config schema for pareto-frontier
---------------------------------------
benchmark: mkt-cnv-160          # used to build paths in shorthand mode
stock: n5-stock                 # used to build paths in shorthand mode
top_k: 10
x_axis: cost                    # cost (default) or time
output_dir: ./6-comparisons/my-run   # relative to the yaml file

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
        color: "#fe7295"
        legend: Ariadne R3
        short: R3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from retrocast.io.blob import load_json_gz
from retrocast.models.stats import ModelStatistics
from retrocast.utils.logging import logger
from retrocast.visualization import plots


def _resolve_statistics_path(entry: dict, source_root: Path | None, benchmark: str, stock: str, yaml_dir: Path) -> Path:
    """Return the statistics.json.gz path for a model entry.

    Explicit ``statistics`` key wins; otherwise build from source root.
    Relative paths are resolved against the yaml file's directory.
    """
    if "statistics" in entry:
        p = Path(entry["statistics"])
        return p if p.is_absolute() else yaml_dir / p

    if source_root is None:
        raise ValueError(f"Model '{entry.get('name')}' must have either 'statistics' or a source-level 'root'")

    return source_root / "5-results" / benchmark / entry["name"] / stock / "statistics.json.gz"


def handle_pareto_frontier(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    yaml_dir = config_path.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    benchmark: str = cfg["benchmark"]
    stock: str = cfg["stock"]
    top_k: int = cfg.get("top_k", 10)
    # CLI --time flag overrides yaml x_axis setting
    time_based: bool = args.time or (cfg.get("x_axis", "cost") == "time")
    auto_open: bool = not args.no_open

    output_dir = yaml_dir / cfg.get("output_dir", "./6-comparisons")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load statistics ---
    stats_list: list[ModelStatistics] = []
    hourly_costs: dict[str, float] = {}
    model_config: dict[str, dict[str, str]] = {}

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

    suffix = "_time" if time_based else ""
    html_path = output_dir / f"pareto_frontier{suffix}.html"
    pdf_path = output_dir / f"pareto_frontier{suffix}.pdf"

    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=auto_open)
    fig.write_image(pdf_path)
    logger.info(f"[bold green]Done![/] Saved to: [underline]{html_path}[/]")
