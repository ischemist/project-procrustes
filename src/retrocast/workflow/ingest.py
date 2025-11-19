from pathlib import Path
from typing import Any

from tqdm import tqdm

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.io.files import save_json_gz
from retrocast.io.manifests import generate_model_hash
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route
from retrocast.utils.logging import logger

# Registry of sampling functions
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
    anonymize: bool = True,
    sampling_strategy: str | None = None,
    sample_k: int | None = None,
) -> tuple[dict[str, list[Route]], Path, dict[str, int]]:
    """
    Converts raw model outputs into standard format.
    """
    logger.info(f"Ingesting results for {model_name} on {benchmark.name}...")

    # 1. Validate Sampling Config
    if sampling_strategy:
        if sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        if sample_k is None:
            raise ValueError("Must provide sample_k when using a sampling strategy")
        sampler_fn = SAMPLING_STRATEGIES[sampling_strategy]
        logger.info(f"Applying sampling: {sampling_strategy} (k={sample_k})")

    processed_routes: dict[str, list[Route]] = {}
    stats = {"n_raw_inputs": 0, "n_targets_matched": 0, "n_routes_generated": 0, "n_routes_saved": 0}

    # 3. Iterate Raw Data
    for target_id, target in tqdm(benchmark.targets.items(), desc="Ingesting"):
        stats["n_raw_inputs"] += 1

        if target_id not in raw_data:
            # Missing prediction (failed run or timeout)
            # We store empty list for solvability denominator
            processed_routes[target_id] = []
            continue

        raw_payload = raw_data[target_id]
        routes = list(adapter.cast(raw_payload, target=target))
        if not routes:
            processed_routes[target_id] = []
            continue

        # 4. Deduplicate & Sample
        unique_routes = deduplicate_routes(routes)

        if sampling_strategy:
            assert sample_k is not None, "sample_k must be provided when using a sampling strategy"
            # Apply the chosen sampling logic (e.g. keep only top 50)
            unique_routes = sampler_fn(unique_routes, sample_k)

        processed_routes[target_id] = unique_routes
        stats["n_routes_generated"] += len(routes)
        stats["n_routes_saved"] += len(unique_routes)

    # 6. Save
    model_hash = generate_model_hash(model_name)
    folder_name = model_hash if anonymize else model_name

    # Structure: data/processed/{benchmark}/{model}/routes.json.gz
    save_path_dir = output_dir / benchmark.name / folder_name
    save_path_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_path_dir / "routes.json.gz"

    save_json_gz(processed_routes, save_file)

    logger.info(f"Saved {stats['n_routes_saved']} routes covering {stats['n_targets_matched']} targets.")
    return processed_routes, save_file, stats
