from __future__ import annotations

import random
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
    rng = random.Random(seed)
    grouped_pools = []
    for pool in pools:
        grouped: dict[G, list[T]] = defaultdict(list)
        for item in pool:
            group = group_fn(item)
            if group in counts:
                grouped[group].append(item)
        grouped_pools.append(grouped)

    selected = []
    for group, target_count in counts.items():
        group_items = []
        for grouped in grouped_pools:
            needed = target_count - len(group_items)
            if needed <= 0:
                break
            available = grouped[group]
            group_items.extend(rng.sample(available, needed) if len(available) > needed else available)
        selected.extend(group_items)
    return selected


def sample_random(items: Sequence[T], n: int, seed: int) -> list[T]:
    if n > len(items):
        raise ValueError(f"cannot sample {n} from {len(items)} items")
    return random.Random(seed).sample(list(items), n)
