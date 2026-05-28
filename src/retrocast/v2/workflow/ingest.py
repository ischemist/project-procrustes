from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from retrocast.v2.adapters.base import Adapter, AdaptMode
from retrocast.v2.models.task import Target, Task
from retrocast.v2.workflow.adapt import adapt_candidates, adapt_routes
from retrocast.v2.workflow.collect import (
    CollectedCandidates,
    CollectedRoutes,
    collect_candidates,
    collect_routes,
)


def ingest_routes(
    raw_payload: Any,
    adapter: Adapter,
    task: Task,
    *,
    mode: AdaptMode = "strict",
) -> CollectedRoutes:
    """Adapt raw planner output into valid routes and collect them by target id."""
    routes = []
    for target, payload, source_key in _target_payloads(raw_payload, task):
        routes.extend(adapt_routes(payload, adapter, mode=mode, target=target, source_key=source_key))
    return collect_routes(routes, task)


def ingest_candidates(
    raw_payload: Any,
    adapter: Adapter,
    task: Task,
    *,
    mode: AdaptMode = "strict",
) -> CollectedCandidates:
    """Adapt raw planner output into candidates and collect them by target id."""
    candidates = []
    for target, payload, source_key in _target_payloads(raw_payload, task):
        candidates.extend(adapt_candidates(payload, adapter, mode=mode, target=target, source_key=source_key))
    return collect_candidates(candidates, task)


def _target_payloads(raw_payload: Any, task: Task) -> Iterator[tuple[Target | None, Any, str | None]]:
    if not isinstance(raw_payload, Mapping):
        target = next(iter(task.targets.values())) if len(task.targets) == 1 else None
        yield target, raw_payload, None
        return

    for target_id, target in task.targets.items():
        if target_id in raw_payload:
            yield target, raw_payload[target_id], target_id
        elif target.smiles in raw_payload:
            yield target, raw_payload[target.smiles], target.smiles
