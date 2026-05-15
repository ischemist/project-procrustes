from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from itertools import chain
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.exceptions import AdapterError, AdapterSchemaError, ChemError, InputError, UnsupportedAdapterFeatureError
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route, RunStatistics, TargetIdentity, TargetInput

logger = logging.getLogger(__name__)


def _target_from_entry(entry: RawRouteEntry) -> TargetIdentity | None:
    if entry.expected_target_smiles is None:
        return None
    return TargetInput(
        id=entry.expected_target_id or entry.source_key or entry.expected_target_smiles,
        smiles=entry.expected_target_smiles,
    )


def _record_adaptation_failure(stats: RunStatistics | None, entry: RawRouteEntry, code: str) -> None:
    if stats is None:
        return
    stats.record_failure(code, target_id=entry.expected_target_id)


def _adapt_entries(
    entries: Iterator[RawRouteEntry],
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    for entry in entries:
        try:
            route = adapter.cast(
                entry.payload,
                ignore_stereo=ignore_stereo,
                expected_target=_target_from_entry(entry),
            )
        except (AdapterError, ChemError) as exc:
            logger.warning(
                "Adapter failed for raw entry %s: %s [%s]",
                entry.expected_target_id or entry.source_key or entry.source_row_index,
                exc,
                exc.code,
            )
            _record_adaptation_failure(stats, entry, exc.code)
            continue

        if stats is not None:
            stats.successful_routes_before_dedup += 1
        yield route


def _iter_route_first_routes(
    raw_data: Any,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    if isinstance(raw_data, Mapping):
        try:
            entry_iter = adapter.iter_raw_entries(raw_data)
            first_entry = next(entry_iter)
        except StopIteration:
            if stats is not None:
                stats.total_routes_in_raw_files += 1
            return
        except (UnsupportedAdapterFeatureError, AdapterSchemaError):
            yield from _iter_keyed_route_first_routes(
                raw_data,
                adapter,
                ignore_stereo=ignore_stereo,
                stats=stats,
            )
            return
        else:
            if stats is not None:
                stats.total_routes_in_raw_files += 1
            yield from _adapt_entries(
                chain([first_entry], entry_iter),
                adapter,
                ignore_stereo=ignore_stereo,
                stats=stats,
            )
            return

    if stats is not None:
        stats.total_routes_in_raw_files += 1
    yield from _adapt_entries(
        adapter.iter_raw_entries(raw_data),
        adapter,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )


def _iter_keyed_route_first_routes(
    raw_data: Mapping[Any, Any],
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    for source_key, payload in raw_data.items():
        if stats is not None:
            stats.total_routes_in_raw_files += 1
        yield from _adapt_entries(
            adapter.iter_raw_entries(payload, source_key=str(source_key)),
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )


def _iter_benchmark_routes(
    raw_data: Any,
    benchmark: BenchmarkSet,
    adapter: BaseAdapter,
    *,
    ignore_stereo: bool,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    if not isinstance(raw_data, Mapping):
        raise InputError(
            "Raw prediction data must be a mapping keyed by target id or target smiles.",
            code="input.invalid_raw_predictions_corpus",
            context={"actual_type": type(raw_data).__name__},
        )

    for target_id, target in benchmark.targets.items():
        matched_key = None
        if target_id in raw_data:
            matched_key = target_id
        elif target.smiles in raw_data:
            matched_key = target.smiles

        if matched_key is None:
            continue

        if stats is not None:
            stats.total_routes_in_raw_files += 1

        payload = raw_data[matched_key]
        entries = adapter.iter_raw_entries(
            payload,
            source_key=str(matched_key),
            expected_target=target,
        )
        yield from _adapt_entries(
            entries,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )


def iter_adapted_routes(
    raw_data: Any,
    adapter: BaseAdapter,
    *,
    benchmark: BenchmarkSet | None = None,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> Iterator[Route]:
    """
    Adapt a raw artifact into canonical Route objects.

    Adapters expose two explicit responsibilities:
    raw artifact traversal via `iter_raw_entries(...)`, and single-route
    normalization via `cast(...)`.
    """
    if benchmark is not None:
        yield from _iter_benchmark_routes(
            raw_data,
            benchmark,
            adapter,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )
        return

    yield from _iter_route_first_routes(
        raw_data,
        adapter,
        ignore_stereo=ignore_stereo,
        stats=stats,
    )


def adapt_route_corpus(
    raw_data: Any,
    adapter: BaseAdapter,
    *,
    benchmark: BenchmarkSet | None = None,
    ignore_stereo: bool = False,
    stats: RunStatistics | None = None,
) -> list[Route]:
    """Materialize a canonical route corpus from a raw artifact."""
    return list(
        iter_adapted_routes(
            raw_data,
            adapter,
            benchmark=benchmark,
            ignore_stereo=ignore_stereo,
            stats=stats,
        )
    )
