"""
Adapt DirectMultiStep (DMS) predictions from pickle format to retrocast format.

This script converts DMS model predictions stored as pickle files into the
standardized retrocast JSON format, mapping targets by their SMILES to target IDs.

Usage:

    uv run scripts/directmultistep/dev-adapt-par-predictions.py
"""

import logging
import pickle
from pathlib import Path

from retrocast.core import create_processing_manifest, process_raw_data
from tqdm import tqdm

from retrocast.adapters.dms_adapter import DMSAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.io import load_smiles_index, save_json
from retrocast.models.chem import TargetIdentity
from retrocast.utils.hashing import generate_file_hash
from retrocast.utils.logging import logger

logger.setLevel(logging.ERROR)

base_dir = Path(__file__).resolve().parents[2]
EVAL_DIR = base_dir / "data" / "evaluations"
PAROUTES_DIR = base_dir / "data" / "paroutes" / "processed"
OUTPUT_DIR = base_dir / "data" / "processed"


def load_dms_predictions_from_pickle(
    raw_path: Path,
    targets_map: dict[str, TargetIdentity],
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    with open(raw_path, "rb") as f:
        predictions_list: list[list[tuple[str, float]]] = pickle.load(f)

    # Convert to {target_id: [routes]} format
    raw_data_per_target: dict[str, list[dict]] = {}
    load_stats = {
        "total_targets_in_pickle": len(predictions_list),
        "empty_targets": 0,
        "mapped_targets": 0,
        "unmapped_targets": 0,
        "total_routes_loaded": 0,
    }

    for predictions in tqdm(predictions_list, desc="Loading targets"):
        if not predictions:
            load_stats["empty_targets"] += 1
            continue

        # Extract target SMILES from the first route
        # Each prediction is a tuple: (route_string, logprob)
        first_route_str = predictions[0][0]
        first_route = eval(first_route_str)
        target_smiles = canonicalize_smiles(first_route["smiles"])

        # Look up target ID from SMILES
        if target_smiles not in targets_map:
            load_stats["unmapped_targets"] += 1
            continue

        load_stats["mapped_targets"] += 1

        # Convert all routes for this target
        routes = []
        for route_str, _logprob in predictions:
            route = eval(route_str)
            routes.append(route)
            load_stats["total_routes_loaded"] += 1

        raw_data_per_target[targets_map[target_smiles].id] = routes

    return raw_data_per_target, load_stats


def adapt_dms_predictions(
    raw_data_path: Path,
    targets_map: dict[str, TargetIdentity],
    output_dir: Path,
    model_name: str,
    dataset_name: str,
) -> None:
    """
    Convert DMS predictions from pickle to retrocast format with full transformation.

    Args:
        pickle_path: Path to the pickle file containing predictions.
        smiles_index_path: Path to the SMILES-to-target-ID index.
        output_dir: Directory where processed results will be saved.
        model_name: Name of the model (used in manifest).
        dataset_name: Name of the dataset (used in manifest).
    """
    # 1. Load data from pickle
    raw_data_per_target, load_stats = load_dms_predictions_from_pickle(raw_path=raw_data_path, targets_map=targets_map)

    if not raw_data_per_target:
        logger.warning("No valid targets found. Skipping.")
        return

    # 2. Process raw data through DMS adapter
    logger.debug(f"\nTransforming {len(raw_data_per_target)} targets through DMS adapter...")
    adapter = DMSAdapter()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = "results.json.gz"
    output_path = output_dir / output_filename

    stats, output_file_hash, routes_content_hash = process_raw_data(
        raw_data_per_target=raw_data_per_target,
        adapter=adapter,
        targets_map=targets_map,
        output_path=output_path,
    )

    # Check if file was actually created
    if stats.final_unique_routes_saved == 0:
        output_filename = None

    # 3. Create and save manifest
    source_files = {raw_data_path.name: generate_file_hash(raw_data_path)}

    manifest = create_processing_manifest(
        model_name=model_name,
        dataset_name=dataset_name,
        source_files=source_files,
        output_file=output_filename,
        stats=stats,
        output_file_hash=output_file_hash,
        routes_content_hash=routes_content_hash,
        model_hash=model_name,  # Use model_name as hash since we're not anonymizing
    )

    # Add loading statistics to manifest
    manifest["loading_statistics"] = load_stats

    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)
    logger.debug(f"Manifest written to {manifest_path}")

    logger.debug("\nProcessing statistics:")
    logger.debug(f"  total_routes_in_raw_files: {stats.total_routes_in_raw_files}")
    logger.debug(f"  routes_failed_transformation: {stats.routes_failed_transformation}")
    logger.debug(f"  final_unique_routes_saved: {stats.final_unique_routes_saved}")
    logger.debug(f"  num_targets_with_at_least_one_route: {len(stats.targets_with_at_least_one_route)}")


def main():
    n1_map = load_smiles_index(PAROUTES_DIR / "n1-routes-smiles-index.json.gz")
    n5_map = load_smiles_index(PAROUTES_DIR / "n5-routes-smiles-index.json.gz")
    logger.debug(f"{len(n1_map)} keys in n1_map, {len(n5_map)} keys in n5_map")
    # we're not merging the two maps because there are ~17 targets that have different routes in n1 and n5

    model_name = "dms-flash-fp16"
    ds = "n5"
    targets_map = n5_map

    adapt_dms_predictions(
        raw_data_path=EVAL_DIR / model_name / ds / f"{ds}_correct_paths_NS2n.pkl",
        targets_map=targets_map,
        output_dir=OUTPUT_DIR / model_name / ds,
        model_name=model_name,
        dataset_name=f"paroutes-{ds}",
    )


if __name__ == "__main__":
    main()
