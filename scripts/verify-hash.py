"""
Verifies the integrity of a processed Ursa benchmark run.

This script can operate in two modes:
1. Single Mode: Verify a specific manifest file.
2. Batch Mode: Discover and verify all runs, with optional filtering by model or dataset.

---
Example Usage:
---

# 1. Verify a single, specific run
uv run scripts/verify-hash.py --manifest data/processed/rs-first-25/ursa-model-4b6418ea/manifest.json

# 2. Verify ALL runs for ALL models and ALL datasets
uv run scripts/verify-hash.py --all-models --all-datasets

# 3. Verify all models but only for a specific dataset
uv run scripts/verify-hash.py --all-models --dataset rs-first-25

# 4. Verify a specific model across all datasets it was run on
uv run scripts/verify-hash.py --model dms-flash-fp16 --all-datasets
"""

import argparse
import json
import sys
from pathlib import Path

from retrocast.exceptions import UrsaException
from retrocast.utils.hashing import generate_file_hash, generate_source_hash
from retrocast.utils.logging import logger

# ANSI color codes for pretty printing
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def verify_single_run(manifest_path: Path, base_dir: Path) -> bool:
    """
    Performs verification for a single manifest file. Returns True on success, False on failure.
    """
    logger.info("-" * 80)
    try:
        logger.info(f"Verifying manifest: {manifest_path}")
        with manifest_path.open("r") as f:
            manifest_data = json.load(f)

        original_hash = manifest_data["source_hash"]
        model_name = manifest_data["model_name"]
        dataset_name = manifest_data["dataset_name"]
        source_files = manifest_data["source_files"]

        raw_dir = base_dir / "data" / "evaluations" / model_name / dataset_name
        logger.info(f"Run details: model='{model_name}', dataset='{dataset_name}'")
        logger.info(f"Expected raw data dir: {raw_dir}")
        logger.info(f"Original source hash: {original_hash}")

        recalculated_hashes = []
        for filename in sorted(source_files.keys()):
            file_path = raw_dir / filename
            if not file_path.is_file():
                logger.error(f"{RED}  -> FAILURE: Source file '{filename}' not found at {file_path}{RESET}")
                return False

            file_hash = generate_file_hash(file_path)
            recalculated_hashes.append(file_hash)

        recalculated_run_hash = generate_source_hash(model_name, recalculated_hashes)
        logger.info(f"Recalculated source hash: {recalculated_run_hash}")

        if recalculated_run_hash == original_hash:
            logger.info(f"{GREEN}  -> SUCCESS: Hashes match!{RESET}")
            return True
        else:
            logger.error(f"{RED}  -> FAILURE: Hashes DO NOT match.{RESET}")
            return False

    except FileNotFoundError:
        logger.error(f"{RED}  -> FAILURE: Manifest file not found at: {manifest_path}{RESET}")
        return False
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"{RED}  -> FAILURE: Manifest is corrupted or malformed. Error: {e}{RESET}")
        return False
    except UrsaException as e:
        logger.error(f"{RED}  -> FAILURE: I/O error reading a source file: {e}{RESET}")
        return False


def main() -> None:
    """Main function to parse arguments and orchestrate verification."""
    parser = argparse.ArgumentParser(
        description="Verify the integrity of processed Ursa benchmark runs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--manifest", type=Path, help="Path to a single manifest.json file to verify.")
    mode_group.add_argument(
        "--all-models", action="store_true", help="Batch mode: discover and verify runs for all models."
    )
    mode_group.add_argument("--model", type=str, help="Batch mode: verify runs for a specific model.")

    parser.add_argument("--dataset", type=str, help="Filter batch mode by a specific dataset name.")
    parser.add_argument("--all-datasets", action="store_true", help="In batch mode, run for all datasets.")

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory of the project. Defaults to current directory.",
    )
    args = parser.parse_args()

    # --- Mode 1: Single Manifest Verification ---
    if args.manifest:
        if args.dataset or args.all_datasets or args.model:
            parser.error("--dataset, --all-datasets, and --model flags are only for batch mode (e.g., --all-models).")
        success = verify_single_run(args.manifest, args.base_dir)
        sys.exit(0 if success else 1)

    # --- Mode 2: Batch Verification ---
    if not (args.dataset or args.all_datasets):
        parser.error("In batch mode (--all-models or --model), you must specify --dataset or --all-datasets.")

    processed_dir = args.base_dir / "data" / "processed"
    all_manifests = list(processed_dir.glob("*/*/manifest.json"))

    if not all_manifests:
        logger.error(f"No manifest files found in {processed_dir}. Nothing to verify.")
        sys.exit(1)

    manifests_to_check = []
    logger.info(f"Discovered {len(all_manifests)} total manifests. Applying filters...")

    for manifest_path in all_manifests:
        try:
            with manifest_path.open("r") as f:
                manifest_data = json.load(f)
            model_name = manifest_data["model_name"]
            dataset_name = manifest_data["dataset_name"]

            model_match = args.all_models or (args.model == model_name)
            dataset_match = args.all_datasets or (args.dataset == dataset_name)

            if model_match and dataset_match:
                manifests_to_check.append(manifest_path)
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"{YELLOW}Skipping corrupted manifest: {manifest_path}{RESET}")
            continue

    if not manifests_to_check:
        logger.error("No runs matched the specified filters.")
        sys.exit(1)

    logger.info(f"Found {len(manifests_to_check)} runs to verify.")
    success_count = 0
    failure_count = 0

    for manifest_path in manifests_to_check:
        if verify_single_run(manifest_path, args.base_dir):
            success_count += 1
        else:
            failure_count += 1

    logger.info("=" * 80)
    logger.info("Verification Summary")
    logger.info("=" * 80)
    logger.info(f"Total runs checked: {len(manifests_to_check)}")
    logger.info(f"{GREEN}Successful: {success_count}{RESET}")
    logger.info(f"{RED}Failed:     {failure_count}{RESET}")

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
