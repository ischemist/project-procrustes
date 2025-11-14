import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.domain.schemas import RunStatistics, TargetInfo
from retrocast.domain.tree import (
    deduplicate_routes,
    sample_k_by_length,
    sample_random_k,
    sample_top_k,
)
from retrocast.exceptions import RetroCastIOError
from retrocast.io import load_json_gz, save_json, save_json_gz
from retrocast.utils.hashing import generate_file_hash, generate_model_hash, generate_source_hash
from retrocast.utils.logging import logger

SAMPLING_STRATEGY_MAP = {
    "top-k": sample_top_k,
    "random-k": sample_random_k,
    "by-length": sample_k_by_length,
}


def process_model_run(
    model_name: str,
    dataset_name: str,
    adapter: BaseAdapter,
    raw_results_file: Path,
    processed_dir: Path,
    targets_map: dict[str, TargetInfo],
    sampling_strategy: str | None = None,
    sample_k: int | None = None,
) -> None:
    """
    Orchestrates the processing pipeline for a model's output.

    Includes optional top-k filtering and organizes results by a stable
    model hash and the dataset name.
    """
    logger.info(f"--- Starting retrocast Processing for Model: '{model_name}' on Dataset: '{dataset_name}' ---")

    # 1. HASHING & PATH SETUP
    model_hash = generate_model_hash(model_name)
    source_hash = generate_source_hash(model_name, [generate_file_hash(raw_results_file)])
    logger.info(f"Stable model hash: '{model_hash}'")
    logger.info(f"Run source hash: '{source_hash}'")

    output_dir = processed_dir / model_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. DATA LOADING & PROCESSING
    final_output_data: dict[str, list[dict[str, Any]]] = {}
    stats = RunStatistics()
    source_file_info = {raw_results_file.name: generate_file_hash(raw_results_file)}

    try:
        logger.info(f"Processing file: {raw_results_file.name}")
        raw_data_per_target = load_json_gz(raw_results_file)
    except RetroCastIOError as e:
        logger.error(f"FATAL: Could not read or parse input file {raw_results_file}. Aborting. Error: {e}")
        return

    pbar = tqdm(raw_data_per_target.items(), desc="Processing targets", unit="target")
    for target_id, raw_routes_list in pbar:
        if target_id not in targets_map:
            logger.warning(f"Skipping routes for '{target_id}': No target info found.")
            continue

        transformed_trees = list(adapter.adapt(raw_routes_list, targets_map[target_id]))

        # Apply filtering based on the chosen strategy
        if sampling_strategy:
            if sample_k is None:
                logger.warning(
                    f"Sampling strategy '{sampling_strategy}' specified but 'sample_k' is not set. Skipping."
                )
            else:
                if sampling_strategy in SAMPLING_STRATEGY_MAP:
                    transformed_trees = SAMPLING_STRATEGY_MAP[sampling_strategy](transformed_trees, sample_k)
                else:
                    logger.warning(f"Unknown sampling strategy '{sampling_strategy}'. Skipping sampling.")

        unique_trees = deduplicate_routes(transformed_trees)

        if len(unique_trees):
            final_output_data[target_id] = [tree.model_dump() for tree in unique_trees]
            stats.targets_with_at_least_one_route.add(target_id)
            stats.routes_per_target[target_id] = len(unique_trees)

        stats.successful_routes_before_dedup += len(transformed_trees)
        stats.final_unique_routes_saved += len(unique_trees)

    # 3. SERIALIZATION & MANIFEST
    if final_output_data:
        output_filename = "results.json.gz"
        output_path = output_dir / output_filename
        logger.info(f"Writing {stats.final_unique_routes_saved} unique routes to: {output_path}")
        save_json_gz(final_output_data, output_path)
    else:
        logger.warning("No routes were successfully processed. No output file written.")
        output_filename = None

    manifest: dict[str, Any] = {
        "model_name": model_name,
        "model_hash": model_hash,
        "dataset_name": dataset_name,
        "source_hash": source_hash,
        "results_file": output_filename,
        "processing_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source_files": source_file_info,
        "statistics": stats.to_manifest_dict(),
    }
    # --- AND HERE: Record the parameter in the manifest if it was used ---
    if sampling_strategy is not None:
        manifest["sampling_parameters"] = {"strategy": sampling_strategy, "k": sample_k}

    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)
    logger.info(f"--- Processing Complete. Manifest written to {manifest_path} ---")
