import random
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def sample_stratified_priority(
    pools: list[list[T]],  # Priority ordered: [n5_pool, n1_pool]
    group_fn: Callable[[T], int],  # Function that takes item and returns group key (e.g. length)
    counts: dict[int, int],
    seed: int,
) -> list[T]:
    """
    Samples items to meet counts, exhausting pools in order.
    """
    # 1. Group all pools individually
    grouped_pools = []
    for pool in pools:
        g = defaultdict(list)
        for item in pool:
            key = group_fn(item)
            if key in counts:
                g[key].append(item)
        grouped_pools.append(g)

    rng = random.Random(seed)
    sampled = []

    for key, target_count in counts.items():
        collected_for_group = []

        # Iterate through pools in priority order
        for group_pool in grouped_pools:
            available = group_pool[key]
            needed = target_count - len(collected_for_group)

            if needed <= 0:
                break

            if len(available) >= needed:
                # We have enough in this pool to finish
                # Sort for stability before sampling
                # (assuming T is not comparable, rely on list order which should be stable from loader)
                selection = rng.sample(available, needed)
                collected_for_group.extend(selection)
            else:
                # Take everything and move to next pool
                collected_for_group.extend(available)

        if len(collected_for_group) < target_count:
            raise ValueError(
                f"Cannot sample {target_count} items for group {key}; "
                f"only found {len(collected_for_group)} across all pools."
            )

        sampled.extend(collected_for_group)

    return sampled


def sample_random(items: list[T], n: int, seed: int) -> list[T]:
    """Simple random sampling."""
    if n > len(items):
        raise ValueError(f"Cannot sample {n} from {len(items)} items.")

    rng = random.Random(seed)
    return rng.sample(items, n)
