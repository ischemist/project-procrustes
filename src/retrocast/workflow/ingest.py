import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.exceptions import InputError
from retrocast.io.data import save_routes
from retrocast.io.provenance import generate_model_hash
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route, RunStatistics
from retrocast.workflow.adapt import adapt_route_corpus
from retrocast.workflow.collect import collect_benchmark_predictions

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
    adapter_name: str | None = None,
) -> tuple[dict[str, list[Route]], Path, RunStatistics]:
    """
    Convert raw model outputs into benchmark-keyed routes.

    The workflow is explicit:
    raw payloads -> canonical Route corpus -> benchmark collection -> routes.json.gz
    """
    logger.info(f"Ingesting results for {model_name} on {benchmark.name}...")

    sampler_fn: Callable[[list[Route], int], list[Route]] | None = None
    sampler_k = 0
    if sampling_strategy is not None:
        if sampling_strategy not in SAMPLING_STRATEGIES:
            raise InputError(
                f"Unknown sampling strategy: {sampling_strategy}",
                code="input.invalid_sampling_strategy",
                context={
                    "sampling_strategy": sampling_strategy,
                    "available_sampling_strategies": sorted(SAMPLING_STRATEGIES.keys()),
                },
            )
        if sample_k is None:
            raise InputError(
                "Must provide sample_k when using a sampling strategy",
                code="input.missing_sample_k",
                context={"sampling_strategy": sampling_strategy},
            )
        sampler_fn = SAMPLING_STRATEGIES[sampling_strategy]
        sampler_k = sample_k
        logger.info(f"Applying sampling: {sampling_strategy} (k={sample_k})")

    stats = RunStatistics()
    route_corpus = adapt_route_corpus(
        raw_data,
        adapter,
        benchmark=benchmark,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )
    collected_routes = collect_benchmark_predictions(route_corpus, benchmark)
    processed_routes = collected_routes.routes_by_target

    if sampler_fn is not None:
        processed_routes = {
            target_id: sampler_fn(routes, sampler_k) if routes else [] for target_id, routes in processed_routes.items()
        }

    stats.final_unique_routes_saved = 0
    stats.targets_with_at_least_one_route.clear()
    stats.routes_per_target.clear()

    for target_id, routes in processed_routes.items():
        stats.final_unique_routes_saved += len(routes)
        if routes:
            stats.targets_with_at_least_one_route.add(target_id)
            stats.routes_per_target[target_id] = len(routes)

    # 6. Save
    model_hash = generate_model_hash(model_name)
    folder_name = model_hash if anonymize else model_name

    save_path_dir = output_dir / benchmark.name / folder_name
    save_path_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_path_dir / "routes.json.gz"

    save_routes(processed_routes, save_file)

    logger.info(
        f"Ingestion complete. Found data for {stats.total_routes_in_raw_files}/{len(benchmark.targets)} targets. "
        f"Adapted {len(route_corpus)} canonical routes. "
        f"Matched {collected_routes.stats.matched_by_canonical_smiles} by smiles. "
        f"Saved {stats.final_unique_routes_saved} valid routes. "
        f"Duplication factor: {stats.duplication_factor}x"
    )
    if collected_routes.stats.unmatched_routes or collected_routes.stats.ambiguous_routes:
        logger.info(
            "Collection outcomes: unmatched=%s ambiguous=%s duplicate_routes_dropped=%s",
            collected_routes.stats.unmatched_routes,
            collected_routes.stats.ambiguous_routes,
            collected_routes.stats.duplicate_routes_dropped,
        )
    if stats.failures_by_code:
        logger.info("Ingestion failures by code: %s", dict(sorted(stats.failures_by_code.items())))

    return processed_routes, save_file, stats
