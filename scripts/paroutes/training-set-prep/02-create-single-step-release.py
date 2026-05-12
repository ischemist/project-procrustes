"""
create public paroutes single-step training-set release files from the released
reaction-holdout route artifact.

usage:
    uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py
    uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py --route-release-dir path/to/reaction-holdout-n1-n5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrocast.curation.training import (
    TrainingReactionReleaseBuilder,
    TrainingSetBuildConfig,
    write_training_reaction_release,
)
from retrocast.io import load_training_route_records
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RELEASE_VERSION = "v2026-05-12"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
DEFAULT_ROUTE_RELEASE_DIR = DEFAULT_RELEASE_ROOT / "reaction-holdout-n1-n5"


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(
        description="create paroutes single-step training-set release files from the reaction-holdout route release."
    )
    parser.add_argument(
        "--route-release-dir",
        type=Path,
        default=DEFAULT_ROUTE_RELEASE_DIR,
        help=f"reaction-holdout route release directory. default: {DEFAULT_ROUTE_RELEASE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help=f"output directory root. default: {DEFAULT_RELEASE_ROOT}",
    )
    args = parser.parse_args()

    route_release_dir = args.route_release_dir
    training_path = route_release_dir / "training.jsonl.gz"
    validation_path = route_release_dir / "validation.jsonl.gz"
    route_manifest_path = route_release_dir / "manifest.json"

    training_routes = load_training_route_records(training_path)
    validation_routes = load_training_route_records(validation_path)
    route_records = [*training_routes, *validation_routes]
    display_route_release_dir = (
        route_release_dir.relative_to(BASE_DIR) if route_release_dir.is_relative_to(BASE_DIR) else route_release_dir
    )
    logger.info("input route release: %s", display_route_release_dir)
    logger.info("  training routes: %s", f"{len(training_routes):,}")
    logger.info("  validation routes: %s", f"{len(validation_routes):,}")

    config = TrainingSetBuildConfig(holdout_mode="reaction", show_progress=False)
    result = TrainingReactionReleaseBuilder(route_records=route_records, config=config).build()
    source_paths = [training_path, validation_path]
    if route_manifest_path.exists():
        source_paths.append(route_manifest_path)

    write_training_reaction_release(
        result=result,
        output_dir=args.output_dir,
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


if __name__ == "__main__":
    main()
