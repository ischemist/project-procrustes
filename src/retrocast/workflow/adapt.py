from __future__ import annotations

from collections.abc import Callable
from itertools import islice
from typing import Any

from pydantic import ValidationError

from retrocast.adapters.base import Adapter, AdaptMode, RawRouteEntry
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterError, ChemError, RetroCastException
from retrocast.models.candidates import Candidate, FailureRecord
from retrocast.models.route import Route
from retrocast.models.task import Target
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr


def adapt_route(
    raw_route_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
) -> Route | None:
    """Adapt one raw route payload into one canonical route, or None on failure."""
    from retrocast import native

    if native.available() and native._adapter_slug(adapter) is not None:
        return native.adapt_route(raw_route_payload, adapter, mode=mode, target=target)
    try:
        return adapter.cast(raw_route_payload, mode=mode, target=target)
    except (AdapterError, ChemError, ValidationError):
        return None


def adapt_routes(
    raw_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
    source_key: str | None = None,
    max_routes: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    workers: int = 1,
) -> list[Route]:
    """Adapt raw planner output into valid canonical routes.

    max_routes counts successful routes. Invalid raw route records are skipped and
    do not consume the limit; use adapt_candidates when raw prediction-slot limits
    and failures must be preserved.
    """
    _validate_limit(max_routes, "max_routes")
    if max_routes == 0:
        return []
    from retrocast import native

    if native._adapter_slug(adapter) is not None:
        candidates = native.adapt(
            raw_payload,
            adapter,
            mode=mode,
            target=target,
            source_key=source_key,
            workers=workers,
        )
        routes = []
        for candidate in candidates:
            if progress_callback is not None:
                progress_callback()
            if candidate.route is None:
                continue
            routes.append(candidate.route)
            if max_routes is not None and len(routes) >= max_routes:
                break
        return routes

    routes: list[Route] = []
    for entry in adapter.iter_raw_routes(raw_payload, source_key=source_key):
        try:
            route = adapt_route(entry.payload, adapter, mode=mode, target=target or _target_from_entry(entry))
            if route is not None:
                routes.append(route)
                if max_routes is not None and len(routes) >= max_routes:
                    break
        finally:
            if progress_callback is not None:
                progress_callback()
    return routes


def adapt_candidates(
    raw_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
    source_key: str | None = None,
    max_candidates: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    workers: int = 1,
) -> list[Candidate]:
    """Adapt raw planner output while preserving failed candidate rank slots."""
    _validate_limit(max_candidates, "max_candidates")
    if max_candidates == 0:
        return []
    from retrocast import native

    if native._adapter_slug(adapter) is not None:
        candidates = native.adapt(
            raw_payload,
            adapter,
            mode=mode,
            target=target,
            source_key=source_key,
            max_candidates=max_candidates,
            workers=workers,
        )
        if progress_callback is not None:
            for _ in candidates:
                progress_callback()
        return candidates

    candidates: list[Candidate] = []
    entries = adapter.iter_raw_routes(raw_payload, source_key=source_key)
    if max_candidates is not None:
        entries = islice(entries, max_candidates)
    for fallback_rank, entry in enumerate(entries, start=1):
        try:
            rank = entry.source_order if entry.source_order is not None else fallback_rank
            entry_target = target or _target_from_entry(entry)
            try:
                route = adapter.cast(entry.payload, mode=mode, target=entry_target)
            except (AdapterError, ChemError, ValidationError) as exc:
                candidates.append(
                    Candidate(rank=rank, failure=_failure_from_exception(exc, target=entry_target, entry=entry))
                )
                continue
            candidates.append(Candidate(rank=rank, route=route))
        finally:
            if progress_callback is not None:
                progress_callback()
    return candidates


def _validate_limit(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _target_from_entry(entry: RawRouteEntry) -> Target | None:
    if entry.target_hint_smiles is None:
        return None

    try:
        smiles = canonicalize_smiles(entry.target_hint_smiles)
        inchikey = get_inchi_key(smiles)
    except ChemError:
        return None
    return Target(
        id=entry.target_hint_id or entry.source_key or smiles,
        smiles=SmilesStr(smiles),
        inchikey=InChIKeyStr(inchikey),
    )


def _failure_from_exception(
    exc: AdapterError | ChemError | ValidationError,
    *,
    target: Target | None,
    entry: RawRouteEntry | None,
) -> FailureRecord:
    """Convert a per-candidate failure into a record that keeps target identity.

    Adapter and chemistry exceptions carry a stable code plus optional structured
    context, e.g. adapter name, bad node type, or offending SMILES. Pydantic
    validation errors do not, so they become generic adapter schema failures.
    """
    code = exc.code if isinstance(exc, RetroCastException) else "adapter.schema_invalid"
    context = dict(exc.context) if isinstance(exc, RetroCastException) else {}
    if target is not None:
        target_id, target_smiles, target_inchikey = target.id, target.smiles, target.inchikey
    else:
        target_id = entry.target_hint_id if entry is not None else None
        target_smiles = target_inchikey = None

    return FailureRecord(
        code=ErrorCode(code),
        message=str(exc),
        target_id=target_id,
        target_smiles=target_smiles,
        target_inchikey=target_inchikey,
        context=context,
    )
