from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from retrocast._warnings import warn_deprecated
from retrocast.curation.filtering import deduplicate_routes
from retrocast.exceptions import BenchmarkCollectionError
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route
from retrocast.models.collections import BenchmarkCollectionStats, CollectedBenchmarkRoutes

UnmatchedPolicy = Literal["ignore", "error", "skip", "report"]
AmbiguousPolicy = Literal["ignore", "error", "skip", "report"]


def _normalize_collection_policy(policy: str, *, name: str) -> Literal["ignore", "error"]:
    if policy == "error":
        return "error"
    if policy == "ignore":
        return "ignore"
    if policy in {"skip", "report"}:
        warn_deprecated(
            old=f"{name}={policy!r}",
            new=f"{name}='ignore'",
            remove_in="0.7",
            note="'skip' and 'report' were aliases for the ignore behavior.",
            stacklevel=3,
        )
        return "ignore"
    raise ValueError(f"{name} must be 'ignore' or 'error'")


def collect_benchmark_predictions(
    routes: Iterable[Route],
    benchmark: BenchmarkSet,
    *,
    on_unmatched: UnmatchedPolicy = "ignore",
    on_ambiguous: AmbiguousPolicy = "error",
    deduplicate: bool = True,
) -> CollectedBenchmarkRoutes:
    """
    Match canonical routes onto a benchmark and return benchmark-keyed routes.

    The core collector is deliberately route-first and benchmark-specific:
    it does not parse raw provider payloads, and it does not require adapter-
    specific provenance in the canonical route corpus.
    """
    stats = BenchmarkCollectionStats()
    routes_by_target: dict[str, list[Route]] = {target_id: [] for target_id in benchmark.targets}
    benchmark_smiles_index: dict[str, list[str]] = defaultdict(list)
    unmatched_policy = _normalize_collection_policy(on_unmatched, name="on_unmatched")
    ambiguous_policy = _normalize_collection_policy(on_ambiguous, name="on_ambiguous")

    for target_id, target in benchmark.targets.items():
        benchmark_smiles_index[target.smiles].append(target_id)

    for route in routes:
        stats.total_routes += 1
        matching_ids = benchmark_smiles_index.get(route.target.smiles, [])

        if len(matching_ids) == 1:
            stats.matched_by_canonical_smiles += 1
            routes_by_target[matching_ids[0]].append(route)
            continue

        if len(matching_ids) > 1:
            stats.ambiguous_routes += 1
            if ambiguous_policy == "error":
                raise BenchmarkCollectionError(
                    "Route matched multiple benchmark targets by canonical smiles.",
                    code="collection.ambiguous_smiles_match",
                    context={"target_smiles": route.target.smiles, "matching_target_ids": matching_ids},
                )
            continue

        stats.unmatched_routes += 1
        if unmatched_policy == "error":
            raise BenchmarkCollectionError(
                "Route could not be matched to the benchmark.",
                code="collection.unmatched_route",
                context={"target_smiles": route.target.smiles},
            )

    collected_routes: dict[str, list[Route]] = {}
    for target_id, target_routes in routes_by_target.items():
        ordered_routes = list(target_routes)
        if deduplicate:
            deduplicated_routes = deduplicate_routes(ordered_routes)
            stats.duplicate_routes_dropped += len(ordered_routes) - len(deduplicated_routes)
            ordered_routes = deduplicated_routes

        collected_routes[target_id] = ordered_routes
        stats.final_unique_routes_saved += len(ordered_routes)

    return CollectedBenchmarkRoutes(
        routes_by_target=collected_routes,
        stats=stats,
    )
