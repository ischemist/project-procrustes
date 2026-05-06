"""
Create public PaRoutes training-set releases.

Usage:
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode route
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode reaction --val-fraction 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from retrocast.curation.training_sets import (
    AdaptedTrainingRoute,
    TrainingHoldoutMode,
    TrainingSetBuildConfig,
    adapt_training_routes,
    build_training_records_from_adapted,
    write_training_release,
)
from retrocast.io import load_raw_paroutes_list, load_stock_file
from retrocast.typing import InchiKeyStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
STOCK_DIR = DATA_DIR / "1-benchmarks" / "stocks"
RELEASE_VERSION = "v2026-05-02"
DEFAULT_OUTPUT_DIR = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION


def build_release(
    mode: TrainingHoldoutMode,
    all_routes: list[AdaptedTrainingRoute],
    raw_all_routes_count: int,
    skipped_all_adaptation: int,
    heldout_routes: dict[str, list[AdaptedTrainingRoute]],
    buyables_stock: set[InchiKeyStr],
    source_paths: list[Path],
    output_dir: Path,
    val_fraction: float,
    seed: int,
    show_progress: bool,
) -> None:
    config = TrainingSetBuildConfig(
        holdout_mode=mode,
        val_fraction=val_fraction,
        seed=seed,
        show_progress=show_progress,
    )

    logger.info(f"Building {config.release_name}...")
    result = build_training_records_from_adapted(
        all_routes=all_routes,
        raw_all_routes_count=raw_all_routes_count,
        skipped_adaptation=skipped_all_adaptation,
        heldout_routes=heldout_routes,
        buyables_stock=buyables_stock,
        config=config,
    )

    logger.info(f"Writing {config.release_name} to {output_dir}...")
    manifest = write_training_release(
        result=result,
        output_dir=output_dir,
        source_paths=source_paths,
        source_root=BASE_DIR,
        parameters={
            "mode": mode,
            "val_fraction": val_fraction,
            "seed": seed,
            "show_progress": show_progress,
            "deduplication": "route.get_signature()",
            "route_holdout": "exclude full n1 union n5 by route.get_signature()",
            "reaction_holdout": "drop routes with any reaction signature present in n1 union n5",
            "buyables_filter": "all route leaves present in buyables-stock by full inchikey",
        },
    )
    logger.info(json.dumps(manifest["statistics"], indent=2, sort_keys=True))


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Create PaRoutes training-set release files.")
    parser.add_argument(
        "--mode",
        choices=["route", "reaction", "both"],
        default="both",
        help="Holdout mode to build. Default: both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction within each stratification bucket. Default: 0.05.",
    )
    parser.add_argument("--seed", type=int, default=20260502, help="Deterministic split seed.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    all_path = RAW_DIR / "all-routes.json.gz"
    n1_path = RAW_DIR / "n1-routes.json.gz"
    n5_path = RAW_DIR / "n5-routes.json.gz"
    buyables_path = STOCK_DIR / "buyables-stock.csv.gz"
    source_paths = [all_path, n1_path, n5_path, buyables_path]

    logger.info("Loading raw PaRoutes files...")
    raw_all_routes = load_raw_paroutes_list(all_path)
    raw_n1_routes = load_raw_paroutes_list(n1_path)
    raw_n5_routes = load_raw_paroutes_list(n5_path)
    buyables_stock = load_stock_file(buyables_path)

    show_progress = not args.no_progress
    collect_reactions = args.mode in ("reaction", "both")
    logger.info("Adapting raw PaRoutes files...")
    all_routes, skipped_all_adaptation = adapt_training_routes(
        raw_all_routes,
        dataset="all",
        id_width=6,
        collect_reactions=collect_reactions,
        show_progress=show_progress,
    )
    n1_routes, _ = adapt_training_routes(
        raw_n1_routes,
        dataset="n1",
        id_width=5,
        collect_reactions=collect_reactions,
        show_progress=show_progress,
    )
    n5_routes, _ = adapt_training_routes(
        raw_n5_routes,
        dataset="n5",
        id_width=5,
        collect_reactions=collect_reactions,
        show_progress=show_progress,
    )

    modes: list[TrainingHoldoutMode] = (
        ["route", "reaction"] if args.mode == "both" else [cast(TrainingHoldoutMode, args.mode)]
    )
    for mode in modes:
        build_release(
            mode=mode,
            all_routes=all_routes,
            raw_all_routes_count=len(raw_all_routes),
            skipped_all_adaptation=skipped_all_adaptation,
            heldout_routes={"n1": n1_routes, "n5": n5_routes},
            buyables_stock=buyables_stock,
            source_paths=source_paths,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            show_progress=show_progress,
        )


if __name__ == "__main__":
    main()
