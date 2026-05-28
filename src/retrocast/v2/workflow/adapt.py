from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterError, ChemError, RetroCastException
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.adapters.base import Adapter, AdaptMode, RawRouteEntry
from retrocast.v2.models.candidates import Candidate, FailureRecord
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target


def adapt_route(
    raw_route_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
) -> Route | None:
    """Adapt one raw route payload into one canonical route, or None on failure."""
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
) -> list[Route]:
    """Adapt raw planner output into valid canonical routes."""
    routes: list[Route] = []
    for entry in adapter.iter_raw_routes(raw_payload, source_key=source_key):
        route = adapt_route(entry.payload, adapter, mode=mode, target=target or _target_from_entry(entry))
        if route is not None:
            routes.append(route)
    return routes


def adapt_candidates(
    raw_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
    source_key: str | None = None,
) -> list[Candidate]:
    """Adapt raw planner output while preserving failed candidate rank slots."""
    candidates: list[Candidate] = []
    next_rank = 1
    for entry in adapter.iter_raw_routes(raw_payload, source_key=source_key):
        rank = entry.source_order if entry.source_order is not None else next_rank
        next_rank += 1
        entry_target = target or _target_from_entry(entry)
        try:
            route = adapter.cast(entry.payload, mode=mode, target=entry_target)
        except (AdapterError, ChemError, ValidationError) as exc:
            candidates.append(
                Candidate(rank=rank, failure=_failure_from_exception(exc, target=entry_target, entry=entry))
            )
            continue
        candidates.append(Candidate(rank=rank, route=route))
    return candidates


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
