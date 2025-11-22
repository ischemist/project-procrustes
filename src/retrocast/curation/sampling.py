import logging
import random
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from retrocast.models.chem import Route

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sample_top_k(routes: list[Route], k: int) -> list[Route]:
    """Keeps the first k routes from the list."""
    if k <= 0:
        return []
    logger.debug(f"Filtering to top {k} routes from {len(routes)}.")
    return routes[:k]


def sample_random_k(routes: list[Route], k: int) -> list[Route]:
    """Keeps a random sample of k routes from the list."""
    if k <= 0:
        return []
    if len(routes) <= k:
        return routes
    logger.debug(f"Randomly sampling {k} routes from {len(routes)}.")
    return random.sample(routes, k)


def sample_k_by_length(routes: list[Route], max_total: int) -> list[Route]:
    """
    Selects up to `max_total` routes by picking one route from each route length
    in a round-robin fashion, starting with the shortest routes.

    This ensures a diverse set of routes biased towards shorter lengths,
    without exceeding the total budget.
    """
    if max_total <= 0:
        return []
    if len(routes) <= max_total:
        return routes

    routes_by_length = defaultdict(list)
    for route in routes:
        routes_by_length[route.length].append(route)

    filtered_routes: list[Route] = []
    sorted_lengths = sorted(routes_by_length.keys())

    level = 0
    while len(filtered_routes) < max_total:
        routes_added_in_pass = 0
        for length in sorted_lengths:
            if level < len(routes_by_length[length]):
                filtered_routes.append(routes_by_length[length][level])
                routes_added_in_pass += 1
                if len(filtered_routes) == max_total:
                    break

        if routes_added_in_pass == 0:
            # No more routes to add from any length group
            break

        if len(filtered_routes) == max_total:
            break

        level += 1

    logger.debug(f"Filtered {len(routes)} routes to {len(filtered_routes)} diverse routes (max total {max_total}).")
    return filtered_routes


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
            logger.warning(
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


SAMPLING_STRATEGIES = {
    "top-k": sample_top_k,
    "random-k": sample_random_k,
    "by-length": sample_k_by_length,
}
