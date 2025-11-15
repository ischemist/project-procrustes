import random
from collections import defaultdict

from retrocast.schemas import Route
from retrocast.utils.logging import logger


def deduplicate_routes(routes: list[Route]) -> list[Route]:
    """
    Filters a list of Route objects, returning only the unique routes.
    Uses the Route.get_signature() method for canonical deduplication.
    """
    seen_signatures = set()
    unique_routes = []

    logger.debug(f"Deduplicating {len(routes)} routes...")

    for route in routes:
        signature = route.get_signature()

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_routes.append(route)

    num_removed = len(routes) - len(unique_routes)
    if num_removed > 0:
        logger.debug(f"Removed {num_removed} duplicate routes.")

    return unique_routes


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


def sample_k_by_depth(routes: list[Route], max_total: int) -> list[Route]:
    """
    Selects up to `max_total` routes by picking one route from each route depth
    in a round-robin fashion, starting with the shortest routes.

    This ensures a diverse set of routes biased towards shorter depths,
    without exceeding the total budget.
    """
    if max_total <= 0:
        return []
    if len(routes) <= max_total:
        return routes

    routes_by_depth = defaultdict(list)
    for route in routes:
        depth = route.depth
        routes_by_depth[depth].append(route)

    filtered_routes: list[Route] = []
    sorted_depths = sorted(routes_by_depth.keys())

    level = 0
    while len(filtered_routes) < max_total:
        routes_added_in_pass = 0
        for depth in sorted_depths:
            if level < len(routes_by_depth[depth]):
                filtered_routes.append(routes_by_depth[depth][level])
                routes_added_in_pass += 1
                if len(filtered_routes) == max_total:
                    break

        if routes_added_in_pass == 0:
            # No more routes to add from any depth group
            break

        if len(filtered_routes) == max_total:
            break

        level += 1

    logger.debug(f"Filtered {len(routes)} routes to {len(filtered_routes)} diverse routes (max total {max_total}).")
    return filtered_routes
