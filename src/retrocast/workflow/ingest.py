import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.io.data import save_routes
from retrocast.io.provenance import generate_model_hash
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route, RunStatistics

logger = logging.getLogger(__name__)

SAMPLING_STRATEGIES = {
    "top-k": sample_top_k,
    "random-k": sample_random_k,
    "by-length": sample_k_by_length,
}


def ingest_model_predictions(
    model_name: str,
    benchmark: BenchmarkSet,
    raw_data: Any,
    adapter: BaseAdapter,
    output_dir: Path,
    anonymize: bool = False,
    sampling_strategy: str | None = None,
    sample_k: int | None = None,
    ignore_stereo: bool = False,
) -> tuple[dict[str, list[Route]], Path, RunStatistics]:
    """
    Converts raw model outputs into standard format.
    Handles raw data keyed by Target ID (preferred) or SMILES (fallback).
    """
    logger.info(f"Ingesting results for {model_name} on {benchmark.name}...")

    if sampling_strategy:
        if sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        if sample_k is None:
            raise ValueError("Must provide sample_k when using a sampling strategy")
        sampler_fn = SAMPLING_STRATEGIES[sampling_strategy]
        logger.info(f"Applying sampling: {sampling_strategy} (k={sample_k})")

    processed_routes: dict[str, list[Route]] = {}

    # Initialize statistics tracking
    stats = RunStatistics()

    # 3. Iterate Benchmark Targets (The Source of Truth)
    for target_id, target in tqdm(benchmark.targets.items(), desc="Ingesting"):
        # --- Resolution Logic ---
        raw_payload = None

        # Strategy A: Direct ID Match (Preferred)
        if target_id in raw_data:
            raw_payload = raw_data[target_id]

        # Strategy B: SMILES Match (Fallback)
        elif target.smiles in raw_data:
            # Warning: If multiple targets have same SMILES, raw_data might be ambiguous.
            # But for ingestion, taking the result for that SMILES is usually correct behavior.
            raw_payload = raw_data[target.smiles]

        if raw_payload is None:
            # Target was not found in the raw predictions
            processed_routes[target_id] = []
            continue

        stats.total_routes_in_raw_files += 1

        # --- Adaptation ---
        try:
            # Adapter returns an iterator of Routes
            routes = list(adapter.cast(raw_payload, target=target, ignore_stereo=ignore_stereo))
        except Exception as e:
            logger.warning(f"Adapter failed for {target_id}: {e}")
            routes = []
            stats.routes_failed_transformation += 1
            processed_routes[target_id] = []
            continue

        if not routes:
            processed_routes[target_id] = []
            continue

        # --- Deduplication & Sampling ---
        unique_routes = deduplicate_routes(routes)

        if sampling_strategy:
            assert sample_k is not None
            unique_routes = sampler_fn(unique_routes, sample_k)

        processed_routes[target_id] = unique_routes

        # --- Stats Update ---
        stats.successful_routes_before_dedup += len(routes)
        stats.final_unique_routes_saved += len(unique_routes)
        if len(unique_routes) > 0:
            stats.targets_with_at_least_one_route.add(target_id)
            stats.routes_per_target[target_id] = len(unique_routes)

    # 6. Save
    model_hash = generate_model_hash(model_name)
    folder_name = model_hash if anonymize else model_name

    save_path_dir = output_dir / benchmark.name / folder_name
    save_path_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_path_dir / "routes.json.gz"

    save_routes(processed_routes, save_file)

    logger.info(
        f"Ingestion complete. Found data for {stats.total_routes_in_raw_files}/{len(benchmark.targets)} targets. "
        f"Saved {stats.final_unique_routes_saved} valid routes. "
        f"Duplication factor: {stats.duplication_factor}x"
    )

    return processed_routes, save_file, stats
