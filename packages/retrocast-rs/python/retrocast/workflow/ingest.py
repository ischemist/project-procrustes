from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from retrocast.adapters.base import Adapter, AdaptMode
from retrocast.models.task import Target, Task
from retrocast.workflow.adapt import adapt_candidates, adapt_routes
from retrocast.workflow.collect import (
    CollectedCandidates,
    CollectedRoutes,
    collect_candidates,
    collect_routes,
)

logger = logging.getLogger(__name__)

_WORKER_ADAPTER: Adapter | None = None
_WORKER_MODE: AdaptMode = "strict"
_WORKER_MAX_CANDIDATES: int | None = None


def ingest_routes(
    raw_payload: Any,
    adapter: Adapter,
    task: Task,
    *,
    mode: AdaptMode = "strict",
    max_routes: int | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> CollectedRoutes:
    """Adapt raw planner output into valid routes and collect them by target id."""
    routes = []
    for target, payload, source_key in _target_payloads(raw_payload, task):
        routes.extend(
            adapt_routes(
                payload,
                adapter,
                mode=mode,
                target=target,
                source_key=source_key,
                max_routes=max_routes,
                progress_callback=progress_callback,
            )
        )
    return collect_routes(routes, task)


def ingest_candidates(
    raw_payload: Any,
    adapter: Adapter,
    task: Task,
    *,
    mode: AdaptMode = "strict",
    max_candidates: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    workers: int = 1,
) -> CollectedCandidates:
    """Adapt raw planner output into candidates and collect them by target id."""
    from retrocast import native

    if native._adapter_slug(adapter) is not None:
        return native.ingest(
            raw_payload,
            adapter,
            task,
            mode=mode,
            max_candidates=max_candidates,
            workers=workers,
        )
    if workers > 1:
        jobs = list(_target_payloads(raw_payload, task))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_ingest_worker,
            initargs=(adapter, mode, max_candidates),
        ) as executor:
            chunks = executor.map(_ingest_target_worker, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
            candidates = [candidate for chunk in chunks for candidate in chunk]
        return collect_candidates(candidates, task)
    candidates = []
    for target, payload, source_key in _target_payloads(raw_payload, task):
        candidates.extend(
            adapt_candidates(
                payload,
                adapter,
                mode=mode,
                target=target,
                source_key=source_key,
                max_candidates=max_candidates,
                progress_callback=progress_callback,
            )
        )
    return collect_candidates(candidates, task)


def _initialize_ingest_worker(adapter: Adapter, mode: AdaptMode, max_candidates: int | None) -> None:
    global _WORKER_ADAPTER, _WORKER_MODE, _WORKER_MAX_CANDIDATES
    _WORKER_ADAPTER = adapter
    _WORKER_MODE = mode
    _WORKER_MAX_CANDIDATES = max_candidates


def _ingest_target_worker(job: tuple[Target | None, Any, str | None]) -> list:
    target, payload, source_key = job
    if _WORKER_ADAPTER is None:
        raise RuntimeError("ingest worker was not initialized")
    return adapt_candidates(
        payload,
        _WORKER_ADAPTER,
        mode=_WORKER_MODE,
        target=target,
        source_key=source_key,
        max_candidates=_WORKER_MAX_CANDIDATES,
    )


def _target_payloads(raw_payload: Any, task: Task) -> Iterator[tuple[Target | None, Any, str | None]]:
    if not isinstance(raw_payload, Mapping):
        if len(task.targets) != 1:
            raise ValueError("Multi-target ingest requires raw_payload keyed by target id or target SMILES.")
        target = next(iter(task.targets.values()))
        yield target, raw_payload, None
        return

    for target_id, target in task.targets.items():
        if target_id in raw_payload:
            yield target, raw_payload[target_id], target_id
        elif target.smiles in raw_payload:
            yield target, raw_payload[target.smiles], target.smiles
        else:
            logger.debug("target %r not found in raw payload by id or smiles; skipping", target_id)
