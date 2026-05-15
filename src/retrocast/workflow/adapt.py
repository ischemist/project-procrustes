from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import ValidationError

from retrocast._warnings import warn_deprecated
from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.exceptions import AdapterError, ChemError, InputError
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route, RunStatistics, TargetIdentity, TargetInput

logger = logging.getLogger(__name__)


def _target_hint_from_entry(entry: RawRouteEntry) -> TargetIdentity | None:
    if entry.target_hint_smiles is None:
        return None
    return TargetInput(
        id=entry.target_hint_id or entry.source_key or entry.target_hint_smiles,
        smiles=entry.target_hint_smiles,
    )


def _record_adaptation_failure(
    stats: RunStatistics | None,
    entry: RawRouteEntry,
    code: str,
    *,
    expected_target: TargetIdentity | None = None,
) -> None:
    if stats is None:
        return
    stats.record_failure(code, target_id=expected_target.id if expected_target is not None else entry.target_hint_id)


def _adapt_entries(
    entries: Iterator[RawRouteEntry],
    adapter: BaseAdapter,
    *,
    expected_target: TargetIdentity | None = None,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    for entry in entries:
        target = expected_target or _target_hint_from_entry(entry)
        try:
            route = adapter.cast(
                entry.payload,
                ignore_stereo=ignore_stereo,
                expected_target=target,
            )
        except ValidationError as exc:
            logger.warning(
                "Adapter failed for raw entry %s: %s [adapter.schema_invalid]",
                (expected_target.id if expected_target is not None else None)
                or entry.target_hint_id
                or entry.source_key
                or entry.source_row_index,
                exc,
            )
            _record_adaptation_failure(stats, entry, "adapter.schema_invalid", expected_target=expected_target)
            continue
        except (AdapterError, ChemError) as exc:
            logger.warning(
                "Adapter failed for raw entry %s: %s [%s]",
                (expected_target.id if expected_target is not None else None)
                or entry.target_hint_id
                or entry.source_key
                or entry.source_row_index,
                exc,
                exc.code,
            )
            _record_adaptation_failure(stats, entry, exc.code, expected_target=expected_target)
            continue

        if stats is not None:
            stats.successful_routes_before_dedup += 1
        yield route


def iter_adapted_routes(
    provider_output: Any,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    if stats is not None:
        stats.total_routes_in_raw_files += 1
    yield from _adapt_entries(
        adapter.iter_raw_entries(provider_output),
        adapter,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )


def iter_target_keyed_adapted_routes(
    target_keyed_provider_output: Mapping[Any, Any],
    benchmark: BenchmarkSet,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    if not isinstance(target_keyed_provider_output, Mapping):
        raise InputError(
            "Target-keyed provider output must be a mapping keyed by target id or target smiles.",
            code="input.invalid_target_keyed_provider_output",
            context={"actual_type": type(target_keyed_provider_output).__name__},
        )

    for target_id, target in benchmark.targets.items():
        matched_key = None
        if target_id in target_keyed_provider_output:
            matched_key = target_id
        elif target.smiles in target_keyed_provider_output:
            matched_key = target.smiles

        if matched_key is None:
            continue

        if stats is not None:
            stats.total_routes_in_raw_files += 1

        payload = target_keyed_provider_output[matched_key]
        entries = adapter.iter_raw_entries(
            payload,
            source_key=str(matched_key),
        )
        yield from _adapt_entries(
            entries,
            adapter,
            expected_target=target,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )


def adapt_target_routes(
    adapter: BaseAdapter,
    raw_target_data: Any,
    target: TargetIdentity,
    *,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    """Adapt a target-local raw payload through the route-first adapter seam."""
    entries = adapter.iter_raw_entries(raw_target_data, source_key=target.id)
    yield from _adapt_entries(
        entries,
        adapter,
        expected_target=target,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )


def adapt_provider_output(
    provider_output: Any,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> list[Route]:
    """Materialize canonical routes from one provider output."""
    return list(iter_adapted_routes(provider_output, adapter, ignore_stereo=ignore_stereo, stats=stats))


def adapt_target_keyed_provider_output(
    target_keyed_provider_output: Mapping[Any, Any],
    benchmark: BenchmarkSet,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> list[Route]:
    """Materialize canonical routes from target-keyed provider output."""
    return list(
        iter_target_keyed_adapted_routes(
            target_keyed_provider_output,
            benchmark,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )
    )


def adapt_route_corpus(
    raw_data: Any,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> list[Route]:
    """Deprecated alias for adapt_provider_output."""
    warn_deprecated(
        old="adapt_route_corpus(...)",
        new="adapt_provider_output(...)",
        remove_in="0.6",
        stacklevel=2,
    )
    return adapt_provider_output(raw_data, adapter, ignore_stereo=ignore_stereo, stats=stats)


def adapt_benchmark_keyed_route_corpus(
    raw_data: Mapping[Any, Any],
    benchmark: BenchmarkSet,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> list[Route]:
    """Deprecated alias for adapt_target_keyed_provider_output."""
    warn_deprecated(
        old="adapt_benchmark_keyed_route_corpus(...)",
        new="adapt_target_keyed_provider_output(...)",
        remove_in="0.6",
        stacklevel=2,
    )
    return adapt_target_keyed_provider_output(
        raw_data,
        benchmark,
        adapter,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )
