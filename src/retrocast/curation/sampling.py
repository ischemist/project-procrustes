import random
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def sample_stratified(
    items: list[T],
    group_fn: Callable[[T], int],  # Function that takes item and returns group key (e.g. length)
    counts: dict[int, int],
    seed: int,
) -> list[T]:
    """
    Samples items to meet specific counts per group.

    Args:
        items: List of objects to sample from.
        group_fn: Function to extract the group key (e.g. lambda x: x.route_length).
        counts: Dict mapping group key to required count (e.g. {2: 100, 3: 100}).
        seed: Random seed for reproducibility.

    Returns:
        List of sampled items.

    Raises:
        ValueError: If not enough items exist for a requested group count.
    """
    # 1. Group items
    grouped = defaultdict(list)
    for item in items:
        key = group_fn(item)
        if key in counts:  # Only store if we care about this group
            grouped[key].append(item)

    # 2. Sample
    rng = random.Random(seed)
    sampled = []

    for key, count in counts.items():
        available = grouped[key]
        if len(available) < count:
            raise ValueError(f"Cannot sample {count} items for group {key}; only {len(available)} available.")

        # Stable sampling: sort first so RNG choice is deterministic given the seed
        # We assume items are comparable or have a consistent order from the input list
        # If input 'items' is stable, this is stable.

        selected = rng.sample(available, count)
        sampled.extend(selected)

    return sampled


def sample_random(items: list[T], n: int, seed: int) -> list[T]:
    """Simple random sampling."""
    if n > len(items):
        raise ValueError(f"Cannot sample {n} from {len(items)} items.")

    rng = random.Random(seed)
    return rng.sample(items, n)
