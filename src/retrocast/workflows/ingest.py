# import datetime
# from pathlib import Path
# from typing import Any

# from tqdm import tqdm

# from retrocast.adapters.base_adapter import BaseAdapter
# from retrocast.domain.tree import (
#     deduplicate_routes,
#     sample_k_by_depth,
#     sample_random_k,
#     sample_top_k,
# )
# from retrocast.exceptions import RetroCastIOError
# from retrocast.io import load_json_gz, save_json, save_json_gz
# from retrocast.models.chem import Route, RunStatistics, TargetIdentity
# from retrocast.utils.hashing import (
#     compute_routes_content_hash,
#     generate_file_hash,
#     generate_model_hash,
#     generate_source_hash,
# )
# from retrocast.utils.logging import logger

# SAMPLING_STRATEGY_MAP = {
#     "top-k": sample_top_k,
#     "random-k": sample_random_k,
#     "by-length": sample_k_by_depth,
# }


# def process_raw_data(
#     raw_data_per_target: dict[str, list[Any]],
#     adapter: BaseAdapter,
#     targets_map: dict[str, TargetIdentity],
#     output_path: Path,
#     sampling_strategy: str | None = None,
#     sample_k: int | None = None,
# ) -> tuple[RunStatistics, str, str]:
#     """
#     Core transformation logic for processing raw model outputs.

#     Handles the transformation of raw route data through an adapter, applies optional
#     sampling strategies, deduplicates results, and saves the output file.

#     Args:
#         raw_data_per_target: Dictionary mapping target IDs to lists of raw route data.
#         adapter: The adapter to use for transforming raw routes.
#         targets_map: Dictionary mapping target IDs to TargetIdentity objects.
#         output_path: Path where the results JSON file will be saved.
#         sampling_strategy: Optional sampling strategy to apply (e.g., "top-k", "random-k", "by-length").
#         sample_k: Number of routes to sample when using a sampling strategy.

#     Returns:
#         A tuple containing:
#         - RunStatistics object with processing statistics.
#         - SHA256 hash of the output file.
#         - Content hash of all routes (order-agnostic).
#     """

#     final_output_data: dict[str, list[dict[str, Any]]] = {}
#     routes_objects: dict[str, list[Route]] = {}
#     stats = RunStatistics()

#     pbar = tqdm(targets_map.items(), desc="Processing targets", unit="target")
#     for _, target_input in pbar:
#         target_id = target_input.id
#         # Get raw routes for this target (if any exist)
#         raw_routes_list = raw_data_per_target.get(target_id, [])

#         # Count total routes in raw input files
#         num_raw_routes = len(raw_routes_list)
#         stats.total_routes_in_raw_files += num_raw_routes

#         # Transform routes through adapter (handles validation and transformation)
#         transformed_trees = list(adapter.cast(raw_routes_list, target_input))

#         # Track failures (both validation and transformation failures)
#         num_failed = num_raw_routes - len(transformed_trees)
#         stats.routes_failed_transformation += num_failed

#         # Apply filtering based on the chosen strategy
#         if sampling_strategy:
#             if sample_k is None:
#                 logger.warning(
#                     f"Sampling strategy '{sampling_strategy}' specified but 'sample_k' is not set. Skipping."
#                 )
#             else:
#                 if sampling_strategy in SAMPLING_STRATEGY_MAP:
#                     transformed_trees = SAMPLING_STRATEGY_MAP[sampling_strategy](transformed_trees, sample_k)
#                 else:
#                     logger.warning(f"Unknown sampling strategy '{sampling_strategy}'. Skipping sampling.")

#         unique_trees = deduplicate_routes(transformed_trees)

#         # Always save an entry for the target (even if empty) to track denominator for solvability
#         if len(unique_trees):
#             final_output_data[target_id] = [
#                 tree.model_dump(mode="json", exclude_computed_fields=True) for tree in unique_trees
#             ]
#             routes_objects[target_id] = unique_trees
#             stats.targets_with_at_least_one_route.add(target_id)
#             stats.routes_per_target[target_id] = len(unique_trees)
#         else:
#             # Save empty list to maintain target in output for correct solvability denominator
#             final_output_data[target_id] = []
#             routes_objects[target_id] = []
#             stats.routes_per_target[target_id] = 0

#         stats.successful_routes_before_dedup += len(transformed_trees)
#         stats.final_unique_routes_saved += len(unique_trees)

#     # Save the output file
#     if final_output_data:
#         logger.info(f"Writing {stats.final_unique_routes_saved} unique routes to: {output_path}")
#         save_json_gz(final_output_data, output_path)

#         # Compute hashes
#         output_file_hash = generate_file_hash(output_path)
#         routes_content_hash = compute_routes_content_hash(routes_objects)
#     else:
#         logger.warning("No routes were successfully processed. No output file written.")
#         output_file_hash = ""
#         routes_content_hash = ""

#     return stats, output_file_hash, routes_content_hash


# def create_processing_manifest(
#     model_name: str,
#     dataset_name: str,
#     source_files: dict[str, str],
#     output_file: str | None,
#     stats: RunStatistics,
#     output_file_hash: str,
#     routes_content_hash: str,
#     sampling_strategy: str | None = None,
#     sample_k: int | None = None,
#     model_hash: str | None = None,
# ) -> dict[str, Any]:
#     """
#     Create a standardized manifest for processed model results.

#     Args:
#         model_name: The name of the model.
#         dataset_name: The name of the dataset.
#         source_files: Dictionary mapping source filenames to their hashes.
#         output_file: Name of the output results file (or None if no results were saved).
#         stats: RunStatistics object with processing statistics.
#         output_file_hash: SHA256 hash of the output results file.
#         routes_content_hash: Content hash of all routes (order-agnostic).
#         sampling_strategy: Optional sampling strategy that was applied.
#         sample_k: Number of routes sampled (when using sampling strategy).
#         model_hash: Optional pre-computed model hash. If not provided, will be generated from model_name.

#     Returns:
#         Dictionary containing the manifest with all metadata.
#     """
#     if model_hash is None:
#         model_hash = generate_model_hash(model_name)

#     source_hash = generate_source_hash(model_name, list(source_files.values()))

#     # Build output_files dict with file hash and content hash
#     output_files: dict[str, Any] = {}
#     if output_file:
#         output_files[output_file] = {
#             "file_hash": output_file_hash,
#             "content_hash": routes_content_hash,
#         }

#     manifest: dict[str, Any] = {
#         "model_name": model_name,
#         "model_hash": model_hash,
#         "dataset_name": dataset_name,
#         "source_hash": source_hash,
#         "processing_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
#         "source_files": source_files,
#         "output_files": output_files,
#         "statistics": stats.to_manifest_dict(),
#     }

#     if sampling_strategy is not None:
#         manifest["sampling_parameters"] = {"strategy": sampling_strategy, "k": sample_k}

#     return manifest


# def process_model_run(
#     model_name: str,
#     dataset_name: str,
#     adapter: BaseAdapter,
#     raw_results_file: Path,
#     processed_dir: Path,
#     targets_map: dict[str, TargetIdentity],
#     sampling_strategy: str | None = None,
#     sample_k: int | None = None,
#     anonymize: bool = True,
# ) -> None:
#     """
#     Orchestrates the processing pipeline for a model's output.

#     This is the high-level orchestration function that combines data loading,
#     transformation, and result serialization with manifest creation.

#     Args:
#         model_name: The name of the model.
#         dataset_name: The name of the dataset.
#         adapter: The adapter to use for transforming raw routes.
#         raw_results_file: Path to the raw results file (gzipped JSON).
#         processed_dir: Directory where processed results will be saved.
#         targets_map: Dictionary mapping target IDs to TargetIdentity objects.
#         sampling_strategy: Optional sampling strategy to apply.
#         sample_k: Number of routes to sample when using a sampling strategy.
#         anonymize: If True, use hashed model name for output directory. If False, use model_name directly.
#     """
#     logger.info(f"--- Starting retrocast Processing for Model: '{model_name}' on Dataset: '{dataset_name}' ---")

#     # 1. HASHING & PATH SETUP
#     model_hash = generate_model_hash(model_name)
#     logger.info(f"Stable model hash: '{model_hash}'")

#     if anonymize:
#         output_subdir = model_hash
#     else:
#         output_subdir = model_name

#     output_dir = processed_dir / output_subdir
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 2. DATA LOADING
#     source_file_info = {raw_results_file.name: generate_file_hash(raw_results_file)}

#     try:
#         logger.info(f"Processing file: {raw_results_file.name}")
#         raw_data_per_target = load_json_gz(raw_results_file)
#     except RetroCastIOError as e:
#         logger.error(f"FATAL: Could not read or parse input file {raw_results_file}. Aborting. Error: {e}")
#         return

#     # 3. PROCESSING & SERIALIZATION
#     output_filename = "results.json.gz"
#     output_path = output_dir / output_filename

#     stats, output_file_hash, routes_content_hash = process_raw_data(
#         raw_data_per_target=raw_data_per_target,
#         adapter=adapter,
#         targets_map=targets_map,
#         output_path=output_path,
#         sampling_strategy=sampling_strategy,
#         sample_k=sample_k,
#     )

#     # 4. MANIFEST
#     # Check if file was actually created (stats.final_unique_routes_saved > 0)
#     if stats.final_unique_routes_saved == 0:
#         output_filename = None

#     manifest = create_processing_manifest(
#         model_name=model_name,
#         dataset_name=dataset_name,
#         source_files=source_file_info,
#         output_file=output_filename,
#         stats=stats,
#         output_file_hash=output_file_hash,
#         routes_content_hash=routes_content_hash,
#         sampling_strategy=sampling_strategy,
#         sample_k=sample_k,
#         model_hash=model_hash,
#     )

#     manifest_path = output_dir / "manifest.json"
#     save_json(manifest, manifest_path)
#     logger.info(f"--- Processing Complete. Manifest written to {manifest_path} ---")
