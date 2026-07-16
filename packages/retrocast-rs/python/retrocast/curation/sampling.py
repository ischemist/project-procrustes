"""Python-facing deterministic sampling backed by Rust's CPython-compatible RNG."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
G = TypeVar("G")


def sample_stratified_priority(
    pools: Sequence[Sequence[T]],
    group_fn: Callable[[T], G],
    counts: dict[G, int],
    seed: int,
) -> list[T]:
    """Sample target counts by group, exhausting pools in priority order."""
    from retrocast import native

    grouped_pools: list[dict[G, list[T]]] = []
    for pool in pools:
        grouped: dict[G, list[T]] = defaultdict(list)
        for item in pool:
            group = group_fn(item)
            if group in counts:
                grouped[group].append(item)
        grouped_pools.append(grouped)

    groups = list(counts)
    coordinates = native.sample_stratified_priority_indices(
        [[len(pool.get(group, ())) for pool in grouped_pools] for group in groups],
        list(counts.values()),
        seed,
    )
    return [
        grouped_pools[pool_index][group][item_index]
        for group, group_coordinates in zip(groups, coordinates, strict=True)
        for pool_index, item_index in group_coordinates
    ]


def sample_random(items: Sequence[T], n: int, seed: int) -> list[T]:
    """Return a deterministic sample using Rust's CPython-compatible RNG."""
    if n > len(items):
        raise ValueError(f"cannot sample {n} from {len(items)} items")
    from retrocast import native

    return [items[index] for index in native.sample_indices(len(items), n, seed)]
