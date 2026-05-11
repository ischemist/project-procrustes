"""
Create public PaRoutes route-training-set releases.

Usage:
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode route
    uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode reaction --val-fraction 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from retrocast.curation.training_sets import (
    TrainingHoldoutMode,
    TrainingSetBuildConfig,
    adapt_training_routes,
    build_training_records_from_adapted,
    write_training_release,
)
from retrocast.io import load_raw_paroutes_list
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
RELEASE_VERSION = "v2026-05-11"
DEFAULT_OUTPUT_DIR = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="Create PaRoutes route-training-set release files.")
    parser.add_argument(
        "--mode",
        choices=["route", "reaction", "both"],
        default="both",
        help="Holdout mode for the route release to build. Default: both.",
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
        help="Disable progress bars.",
    )
    args = parser.parse_args()

    all_path = RAW_DIR / "all-routes.json.gz"
    n1_path = RAW_DIR / "n1-routes.json.gz"
    n5_path = RAW_DIR / "n5-routes.json.gz"
    source_paths = [all_path, n1_path, n5_path]

    show_progress = not args.no_progress
    raw_all_routes = load_raw_paroutes_list(all_path)
    raw_n1_routes = load_raw_paroutes_list(n1_path)
    raw_n5_routes = load_raw_paroutes_list(n5_path)
    logger.info("Inputs loaded:")
    logger.info("  %s: %s routes", all_path.relative_to(BASE_DIR), f"{len(raw_all_routes):,}")
    logger.info("  %s: %s routes", n1_path.relative_to(BASE_DIR), f"{len(raw_n1_routes):,}")
    logger.info("  %s: %s routes", n5_path.relative_to(BASE_DIR), f"{len(raw_n5_routes):,}")
    collect_reactions = args.mode in ("reaction", "both")
    all_routes, all_adaptation = adapt_training_routes(
        raw_all_routes,
        dataset="all",
        id_width=6,
        collect_reactions=collect_reactions,
        show_progress=show_progress,
    )
    n1_routes, n1_adaptation = adapt_training_routes(
        raw_n1_routes,
        dataset="n1",
        id_width=5,
        collect_reactions=collect_reactions,
        show_progress=show_progress,
    )
    n5_routes, n5_adaptation = adapt_training_routes(
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
        config = TrainingSetBuildConfig(
            holdout_mode=mode,
            val_fraction=args.val_fraction,
            seed=args.seed,
            show_progress=show_progress,
        )

        result = build_training_records_from_adapted(
            all_routes=all_routes,
            all_adaptation=all_adaptation,
            heldout_routes={"n1": n1_routes, "n5": n5_routes},
            heldout_adaptation={"n1": n1_adaptation, "n5": n5_adaptation},
            config=config,
        )
        write_training_release(
            result=result,
            output_dir=args.output_dir,
            source_paths=source_paths,
            source_root=BASE_DIR,
            config=config,
        )
        output = result.summary["output"]
        all_records = output["all_records"]
        postprocessing = result.summary["postprocessing"]
        logger.info(
            "Output: %s records (%s training, %s validation).",
            all_records["total"],
            all_records["training"],
            all_records["validation"],
        )
        logger.info(
            "Postprocessing: %s exact route matches removed; %s duplicates removed.",
            postprocessing["exact_route_matches_removed"],
            postprocessing["duplicate_routes_removed"],
        )
        logger.info(
            "Dedup breakdown: %s exact-chemistry duplicates removed; %s mapped-smiles variants collapsed.",
            postprocessing["chemical_duplicates_removed"],
            postprocessing["mapped_smiles_variants_collapsed"],
        )
        reaction_overlap = postprocessing.get("reaction_overlap")
        if reaction_overlap is not None:
            logger.info(
                "Reaction overlap: %s routes had overlapping reactions; %s fragments kept after excision "
                "(%s routes fully removed).",
                reaction_overlap["routes_with_overlapping_reactions"],
                reaction_overlap["fragments_kept_after_excision"],
                reaction_overlap["routes_fully_removed_after_excision"],
            )
        failures_by_code = result.summary["adaptation"]["all_routes"]["failures_by_code"]
        if failures_by_code:
            logger.info("Adaptation failures by code: %s", failures_by_code)


if __name__ == "__main__":
    main()
