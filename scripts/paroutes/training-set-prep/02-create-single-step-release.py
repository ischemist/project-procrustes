"""
create public paroutes single-step training-set release files from the released
route artifacts.

usage:
    uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py
    uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py --route-release-dir path/to/reaction-holdout-n1-n5
    uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py --mode route
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from retrocast.cli.progress import step_progress
from retrocast.curation.training import (
    TrainingHoldoutMode,
    TrainingReactionReleaseBuilder,
    TrainingSetBuildConfig,
    write_training_reaction_release,
)
from retrocast.io import load_training_route_records
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RELEASE_VERSION = "v2026-05-29"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
DEFAULT_HOLDOUT_MODES: tuple[TrainingHoldoutMode, ...] = ("route", "reaction")


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(
        description="create paroutes single-step training-set release files from released route artifacts."
    )
    parser.add_argument(
        "--route-release-dir",
        type=Path,
        default=None,
        help="explicit route release directory. when omitted, builds from <output-dir>/<mode>-holdout-n1-n5.",
    )
    parser.add_argument(
        "--mode",
        choices=[*DEFAULT_HOLDOUT_MODES, "both"],
        default="both",
        help="route holdout mode to flatten. default: both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help=f"output directory root. default: {DEFAULT_RELEASE_ROOT}",
    )
    args = parser.parse_args()

    modes: list[TrainingHoldoutMode] = list(DEFAULT_HOLDOUT_MODES) if args.mode == "both" else [args.mode]
    if args.route_release_dir is not None and len(modes) != 1:
        parser.error("--route-release-dir requires --mode route or --mode reaction")

    with step_progress(console=Console(), total=3 * len(modes), transient=True) as step:
        for mode in modes:
            route_release_dir = args.route_release_dir or args.output_dir / f"{mode}-holdout-n1-n5"
            build_single_step_release(
                route_release_dir=route_release_dir, output_dir=args.output_dir, mode=mode, step=step
            )


def build_single_step_release(*, route_release_dir: Path, output_dir: Path, mode: TrainingHoldoutMode, step) -> None:
    training_path = route_release_dir / "training.jsonl.gz"
    validation_path = route_release_dir / "validation.jsonl.gz"
    route_manifest_path = route_release_dir / "manifest.json"

    release_name = f"single-step-{mode}-holdout-n1-n5"
    with step(f"{release_name}: load route records"):
        training_routes = load_training_route_records(training_path)
        validation_routes = load_training_route_records(validation_path)
    route_records = [*training_routes, *validation_routes]
    display_route_release_dir = (
        route_release_dir.relative_to(BASE_DIR) if route_release_dir.is_relative_to(BASE_DIR) else route_release_dir
    )
    logger.info("input route release: %s", display_route_release_dir)
    logger.info("  training routes: %s", f"{len(training_routes):,}")
    logger.info("  validation routes: %s", f"{len(validation_routes):,}")

    config = TrainingSetBuildConfig(holdout_mode=mode, show_progress=False)
    with step(f"{release_name}: build reactions"):
        result = TrainingReactionReleaseBuilder(route_records=route_records, config=config).build()
    source_paths = [training_path, validation_path]
    if route_manifest_path.exists():
        source_paths.append(route_manifest_path)

    with step(f"{release_name}: write artifacts"):
        write_training_reaction_release(
            result=result,
            output_dir=output_dir,
            source_paths=source_paths,
            source_root=BASE_DIR,
            config=config,
        )

    output = result.summary["output"]["all_records"]
    reaction_postprocessing = result.summary["reaction_postprocessing"]
    training_postprocessing = reaction_postprocessing["training"]
    validation_postprocessing = reaction_postprocessing["validation"]
    overlap_before_cleanup = reaction_postprocessing["cross_split_overlap_before_cleanup"]
    overlap_after_cleanup = reaction_postprocessing["cross_split_overlap_after_cleanup"]

    logger.info(
        "output: %s reactions (%s training, %s validation).",
        output["total"],
        output["training"],
        output["validation"],
    )
    logger.info(
        "training split: %s flattened; %s exact duplicates removed; %s mapping variants collapsed.",
        training_postprocessing["flattened_reactions"],
        training_postprocessing["chemical_duplicates_removed"],
        training_postprocessing["mapped_smiles_variants_collapsed"],
    )
    logger.info(
        "validation split: %s flattened; %s exact duplicates removed; %s mapping variants collapsed; "
        "%s overlaps removed from validation.",
        validation_postprocessing["flattened_reactions"],
        validation_postprocessing["chemical_duplicates_removed"],
        validation_postprocessing["mapped_smiles_variants_collapsed"],
        validation_postprocessing["overlap_removed_from_validation"],
    )
    logger.info(
        "cross-split overlap before cleanup: %s shared reaction identities; %s shared exact mapped reactions "
        "(training records affected: %s, validation records affected: %s).",
        overlap_before_cleanup["shared_reaction_identities"],
        overlap_before_cleanup["shared_exact_reaction_signatures"],
        overlap_before_cleanup["training_records_with_shared_identity"],
        overlap_before_cleanup["validation_records_with_shared_identity"],
    )
    logger.info(
        "cross-split overlap after cleanup: %s shared reaction identities; %s shared exact mapped reactions.",
        overlap_after_cleanup["shared_reaction_identities"],
        overlap_after_cleanup["shared_exact_reaction_signatures"],
    )
    if mode == "route" and overlap_after_cleanup["shared_reaction_identities"]:
        logger.warning(
            "single-step route-holdout release keeps %s cross-split shared reaction identities "
            "(%s exact mapped reaction signatures).",
            overlap_after_cleanup["shared_reaction_identities"],
            overlap_after_cleanup["shared_exact_reaction_signatures"],
        )


if __name__ == "__main__":
    main()
