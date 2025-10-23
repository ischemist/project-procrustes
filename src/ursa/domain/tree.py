import hashlib
import random
from collections import defaultdict

from ursa.domain.schemas import BenchmarkTree, MoleculeNode
from ursa.utils.logging import logger


def _generate_tree_signature(node: MoleculeNode) -> str:
    """
    Recursively generates a canonical, order-invariant signature for a
    molecule node and its entire history.

    This version does not use memoization as tree objects should not be large
    enough to make this a performance bottleneck, and it avoids subtle bugs.

    Args:
        node: The MoleculeNode to generate a signature for.

    Returns:
        A hash representing the canonical signature of the tree/subtree.
    """
    # Base Case: The node is a starting material. Its signature is its own hash.
    if node.is_starting_material:
        return node.molecule_hash

    # Recursive Step: The node is an intermediate.
    # Its signature depends on the sorted signatures of its reactants.
    if not node.reactions:  # Should not happen with validation, but good to be safe
        return node.molecule_hash

    reactant_signatures = []
    # Assuming one reaction per node as per our schema
    for reactant_node in node.reactions[0].reactants:
        reactant_signatures.append(_generate_tree_signature(reactant_node))

    # Sort the signatures to ensure order-invariance (A.B>>C is same as B.A>>C)
    sorted_signatures = sorted(reactant_signatures)

    # The final signature string incorporates the history and the result.
    signature_string = "".join(sorted_signatures) + ">>" + node.molecule_hash
    signature_bytes = signature_string.encode("utf-8")

    # Hash the canonical representation to get the final signature
    return f"tree_sha256:{hashlib.sha256(signature_bytes).hexdigest()}"


def deduplicate_routes(routes: list[BenchmarkTree]) -> list[BenchmarkTree]:
    """
    Filters a list of BenchmarkTree objects, returning only the unique routes.
    """
    seen_signatures = set()
    unique_routes = []

    logger.debug(f"Deduplicating {len(routes)} routes...")

    for route in routes:
        signature = _generate_tree_signature(route.retrosynthetic_tree)

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_routes.append(route)

    num_removed = len(routes) - len(unique_routes)
    if num_removed > 0:
        logger.debug(f"Removed {num_removed} duplicate routes.")

    return unique_routes


def calculate_route_length(node: MoleculeNode) -> int:
    """
    Calculate the length of a route (number of reactions/steps).

    This counts the number of reactions in the longest path from the target
    to any starting material.
    """
    if node.is_starting_material:
        return 0

    if not node.reactions:
        return 0

    # Find the longest path among all reactants
    max_length = 0
    for reaction in node.reactions:
        for reactant in reaction.reactants:
            reactant_length = calculate_route_length(reactant)
            max_length = max(max_length, reactant_length)

    return max_length + 1


def sample_top_k(routes: list[BenchmarkTree], k: int) -> list[BenchmarkTree]:
    """Keeps the first k routes from the list."""
    if k <= 0:
        return []
    logger.debug(f"Filtering to top {k} routes from {len(routes)}.")
    return routes[:k]


def sample_random_k(routes: list[BenchmarkTree], k: int) -> list[BenchmarkTree]:
    """Keeps a random sample of k routes from the list."""
    if k <= 0:
        return []
    if len(routes) <= k:
        return routes
    logger.debug(f"Randomly sampling {k} routes from {len(routes)}.")
    return random.sample(routes, k)


def sample_k_by_length(routes: list[BenchmarkTree], max_total: int) -> list[BenchmarkTree]:
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
        length = calculate_route_length(route.retrosynthetic_tree)
        routes_by_length[length].append(route)

    filtered_routes: list[BenchmarkTree] = []
    sorted_lengths = sorted(routes_by_length.keys())

    depth = 0
    while len(filtered_routes) < max_total:
        routes_added_in_pass = 0
        for length in sorted_lengths:
            if depth < len(routes_by_length[length]):
                filtered_routes.append(routes_by_length[length][depth])
                routes_added_in_pass += 1
                if len(filtered_routes) == max_total:
                    break

        if routes_added_in_pass == 0:
            # No more routes to add from any length group
            break

        if len(filtered_routes) == max_total:
            break

        depth += 1

    logger.debug(f"Filtered {len(routes)} routes to {len(filtered_routes)} diverse routes (max total {max_total}).")
    return filtered_routes
