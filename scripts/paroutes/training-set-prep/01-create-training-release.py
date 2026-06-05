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

from rich.console import Console

from retrocast.cli.progress import step_progress
from retrocast.curation.training.records import (
    TrainingHoldoutMode,
    TrainingSetBuildConfig,
)
from retrocast.curation.training.route_release import (
    TrainingRouteReleaseBuilder,
    adapt_training_routes,
    write_training_release,
)
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
RELEASE_VERSION = "v2026-06-05"
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
    args = parser.parse_args()

    all_path = RAW_DIR / "all-routes.json.gz"
    n1_path = RAW_DIR / "n1-routes.json.gz"
    n5_path = RAW_DIR / "n5-routes.json.gz"
    source_paths = [all_path, n1_path, n5_path]

    with step_progress(console=Console(), total=3, transient=True) as step:
        with step("all: adapt routes"):
            all_adaptation = adapt_training_routes(all_path, dataset="all", show_progress=False)
        with step("n1: adapt routes"):
            n1_adaptation = adapt_training_routes(n1_path, dataset="n1", show_progress=False)
        with step("n5: adapt routes"):
            n5_adaptation = adapt_training_routes(n5_path, dataset="n5", show_progress=False)
    logger.info("Inputs adapted:")
    logger.info("  %s: %s routes", all_path.relative_to(BASE_DIR), f"{all_adaptation.stats.adapted_routes:,}")
    logger.info("  %s: %s routes", n1_path.relative_to(BASE_DIR), f"{n1_adaptation.stats.adapted_routes:,}")
    logger.info("  %s: %s routes", n5_path.relative_to(BASE_DIR), f"{n5_adaptation.stats.adapted_routes:,}")

    modes: list[TrainingHoldoutMode] = (
        ["route", "reaction"] if args.mode == "both" else [cast(TrainingHoldoutMode, args.mode)]
    )
    for mode in modes:
        config = TrainingSetBuildConfig(
            holdout_mode=mode,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

        result = TrainingRouteReleaseBuilder(
            all_routes=all_adaptation.routes,
            all_adaptation=all_adaptation.stats,
            holdout_routes={"n1": n1_adaptation.routes, "n5": n5_adaptation.routes},
            holdout_adaptation={"n1": n1_adaptation.stats, "n5": n5_adaptation.stats},
            config=config,
        ).build()
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
        release_name = result.release_name
        logger.info(
            "%s output: %s records (%s training, %s validation).",
            release_name,
            all_records["total"],
            all_records["training"],
            all_records["validation"],
        )
        logger.info(
            "%s postprocessing: %s exact route matches removed; %s duplicates removed.",
            release_name,
            postprocessing["exact_route_matches_removed"],
            postprocessing["duplicate_routes_removed"],
        )
        logger.info(
            "%s dedup breakdown: %s exact-chemistry duplicates removed; %s mapped-smiles variants collapsed.",
            release_name,
            postprocessing["chemical_duplicates_removed"],
            postprocessing["mapped_smiles_variants_collapsed"],
        )
        reaction_overlap = postprocessing.get("reaction_overlap")
        if reaction_overlap is not None:
            logger.info(
                "%s reaction overlap: %s routes had overlapping reactions; %s fragments kept after excision "
                "(%s routes fully removed).",
                release_name,
                reaction_overlap["routes_with_overlapping_reactions"],
                reaction_overlap["fragments_kept_after_excision"],
                reaction_overlap["routes_fully_removed_after_excision"],
            )
        failures_by_code = result.summary["adaptation"]["all_routes"]["failures_by_code"]
        if failures_by_code:
            logger.info("%s adaptation failures by code: %s", release_name, failures_by_code)


if __name__ == "__main__":
    main()
