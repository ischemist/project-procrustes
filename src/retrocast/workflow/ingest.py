import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.exceptions import AdapterError, ChemError, InputError
from retrocast.io.data import save_candidate_records, save_routes
from retrocast.io.provenance import generate_model_hash
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.candidates import CandidateAuditMetadata, CandidateRecord, CandidateRecordsDict, CandidateSource
from retrocast.models.chem import Route, RunStatistics
from retrocast.models.validity import FailureRecord
from retrocast.workflow.adapt import adapt_provider_output, adapt_target_keyed_provider_output
from retrocast.workflow.collect import collect_benchmark_predictions

logger = logging.getLogger(__name__)

SAMPLING_STRATEGIES = {
    "top-k": sample_top_k,
    "random-k": sample_random_k,
    "by-length": sample_k_by_length,
}

ProviderOutputKind = Literal["provider_output", "target_keyed_provider_output"]


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
    provider_output_kind: ProviderOutputKind = "target_keyed_provider_output",
    preserve_failed_candidates: bool = False,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[dict[str, list[Route]], Path, RunStatistics]:
    """
    Convert raw model outputs into benchmark-keyed routes.

    The workflow is explicit:
    raw payloads -> PredictedRoute corpus -> benchmark collection -> routes.json.gz
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
    candidate_records: CandidateRecordsDict | None = None
    candidate_metadata: CandidateAuditMetadata | None = None
    if preserve_failed_candidates:
        if provider_output_kind != "target_keyed_provider_output":
            raise InputError(
                "Candidate-preserving ingest currently requires target-keyed provider output.",
                code="input.candidate_preservation_requires_target_keyed_output",
                context={"provider_output_kind": provider_output_kind},
            )
        route_corpus, candidate_records, candidate_metadata = _adapt_target_keyed_candidate_records(
            raw_data,
            benchmark,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
            progress_callback=progress_callback,
        )
    elif provider_output_kind == "target_keyed_provider_output":
        route_corpus = adapt_target_keyed_provider_output(
            raw_data,
            benchmark,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
            progress_callback=progress_callback,
        )
    elif provider_output_kind == "provider_output":
        route_corpus = adapt_provider_output(
            raw_data,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
            progress_callback=progress_callback,
        )
    else:
        raise InputError(
            f"Unknown provider output kind: {provider_output_kind}",
            code="input.invalid_provider_output_kind",
            context={
                "provider_output_kind": provider_output_kind,
                "available_provider_output_kinds": ["provider_output", "target_keyed_provider_output"],
            },
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
    if candidate_records is not None and candidate_metadata is not None:
        save_candidate_records(candidate_records, save_path_dir / "candidates.json.gz", candidate_metadata)

    logger.info(
        f"Ingestion complete. Adapted {stats.total_routes_in_raw_files} raw route entries. "
        f"Adapted {len(route_corpus)} predicted routes. "
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


def _adapt_target_keyed_candidate_records(
    raw_data: Any,
    benchmark: BenchmarkSet,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics,
    progress_callback: Callable[[], None] | None,
) -> tuple[list[Route], CandidateRecordsDict, CandidateAuditMetadata]:
    if not isinstance(raw_data, Mapping):
        raise InputError(
            "Target-keyed provider output must be a mapping keyed by target id or target smiles.",
            code="input.invalid_target_keyed_provider_output",
            context={"actual_type": type(raw_data).__name__},
        )

    route_corpus: list[Route] = []
    records: CandidateRecordsDict = {target_id: [] for target_id in benchmark.targets}
    for target_id, target in benchmark.targets.items():
        matched_key = target_id if target_id in raw_data else target.smiles if target.smiles in raw_data else None
        if matched_key is None:
            continue

        for rank, entry in enumerate(
            adapter.iter_raw_entries(raw_data[matched_key], source_key=str(matched_key)), start=1
        ):
            stats.total_routes_in_raw_files += 1
            if progress_callback is not None:
                progress_callback()
            source = CandidateSource(
                key=entry.source_key,
                row_index=entry.source_row_index,
                record_id=entry.source_record_id,
            )
            try:
                route = adapter.cast(entry.payload, ignore_stereo=ignore_stereo, expected_target=target)
            except (AdapterError, ChemError) as exc:
                stats.record_failure(exc.code, target_id=target_id)
                records[target_id].append(
                    CandidateRecord(
                        rank=rank,
                        adapter_failure=FailureRecord.from_exception(exc),
                        source=source,
                    )
                )
                continue
            except ValidationError as exc:
                stats.record_failure("adapter.schema_invalid", target_id=target_id)
                records[target_id].append(
                    CandidateRecord(
                        rank=rank,
                        adapter_failure=FailureRecord(
                            code="adapter.schema_invalid",
                            message=str(exc),
                        ),
                        source=source,
                    )
                )
                continue

            stats.successful_routes_before_dedup += 1
            route_corpus.append(route)
            records[target_id].append(CandidateRecord(rank=rank, route=route, source=source))

    n_records = sum(len(target_records) for target_records in records.values())
    metadata = CandidateAuditMetadata(
        preserves_failed_candidates=True,
        candidate_denominator="complete",
        n_raw_entries_seen=stats.total_routes_in_raw_files,
        n_candidate_records_written=n_records,
        n_routes_adapted=stats.successful_routes_before_dedup,
        n_adaptation_failures=stats.routes_failed_transformation,
    )
    return route_corpus, records, metadata
