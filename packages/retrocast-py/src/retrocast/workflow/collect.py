from __future__ import annotations

from collections.abc import Iterable

from retrocast.models.candidates import Candidate
from retrocast.models.route import Route
from retrocast.models.task import Task

CollectedCandidates = dict[str, list[Candidate]]
CollectedRoutes = dict[str, list[Route]]


def collect_candidates(candidates: Iterable[Candidate], task: Task) -> CollectedCandidates:
    """Map adapted candidates onto task targets by route or failure identity."""
    collected: CollectedCandidates = {target_id: [] for target_id in task.targets}
    target_id_by_inchikey = {target.inchikey: target_id for target_id, target in task.targets.items()}

    for candidate in candidates:
        target_id = None
        if candidate.route is not None:
            target_id = target_id_by_inchikey.get(candidate.route.target.inchikey)
        elif candidate.failure is not None:
            if candidate.failure.target_id in task.targets:
                target_id = candidate.failure.target_id
            elif candidate.failure.target_inchikey is not None:
                target_id = target_id_by_inchikey.get(candidate.failure.target_inchikey)

        if target_id is not None:
            collected[target_id].append(candidate)
    return collected


def collect_routes(routes: Iterable[Route], task: Task) -> CollectedRoutes:
    """Map valid canonical routes onto task targets."""
    collected: CollectedRoutes = {target_id: [] for target_id in task.targets}
    target_id_by_inchikey = {target.inchikey: target_id for target_id, target in task.targets.items()}

    for route in routes:
        target_id = target_id_by_inchikey.get(route.target.inchikey)
        if target_id is not None:
            collected[target_id].append(route)
    return collected
