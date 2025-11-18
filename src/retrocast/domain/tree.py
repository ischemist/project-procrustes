import random
from collections import defaultdict

from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route
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


def excise_reactions_from_route(
    route: Route,
    exclude: set[ReactionSignature],
) -> list[Route]:
    """
    Remove specific reactions from a route, yielding sub-routes.

    When a reaction is in the exclusion set, the product of that reaction becomes
    a leaf node in its parent tree. Each reactant of the excised reaction that has
    its own synthesis path becomes the root of a new sub-route.

    Args:
        route: The route to process.
        exclude: Set of ReactionSignatures to remove from the route.

    Returns:
        List of valid sub-routes (routes with length > 0) after excision.
        The main route (if still valid) is first, followed by any sub-routes
        created from excised reactants.

    Example:
        Given route: R <- A1 <- A2 <- I1 <- I2 <- A3
        Excluding: I1 <- I2 (i.e., the reaction where I2 produces I1)
        Result:
        - Main route: R <- A1 <- A2 <- I1 (I1 is now a leaf)
        - Sub-route: I2 <- A3 (I2 becomes a new target)
    """
    if route.target.is_leaf:
        # No reactions to excise
        return []

    sub_routes: list[Route] = []

    def _rebuild(node: Molecule) -> Molecule:
        """Recursively rebuild tree, cutting at excluded reactions."""
        if node.is_leaf:
            # Leaf nodes are returned as-is
            return Molecule(
                smiles=node.smiles,
                inchikey=node.inchikey,
                metadata=node.metadata.copy(),
            )

        # Non-leaf node has a synthesis_step
        assert node.synthesis_step is not None

        rxn = node.synthesis_step
        sig: ReactionSignature = (
            frozenset(r.inchikey for r in rxn.reactants),
            node.inchikey,
        )

        if sig in exclude:
            # Cut here: this node becomes a leaf
            # Each non-leaf reactant becomes a new sub-route
            for reactant in rxn.reactants:
                if not reactant.is_leaf:
                    rebuilt_reactant = _rebuild(reactant)
                    if rebuilt_reactant.synthesis_step is not None:
                        # Reactant still has reactions, create sub-route
                        new_route = Route(
                            target=rebuilt_reactant,
                            rank=route.rank,
                            metadata=route.metadata.copy(),
                        )
                        sub_routes.append(new_route)

            # Return this node as a leaf (no synthesis_step)
            return Molecule(
                smiles=node.smiles,
                inchikey=node.inchikey,
                metadata=node.metadata.copy(),
            )
        else:
            # Keep this reaction, but recursively check reactants
            new_reactants = [_rebuild(r) for r in rxn.reactants]
            new_step = ReactionStep(
                reactants=new_reactants,
                mapped_smiles=rxn.mapped_smiles,
                template=rxn.template,
                reagents=rxn.reagents,
                solvents=rxn.solvents,
                metadata=rxn.metadata.copy(),
            )
            return Molecule(
                smiles=node.smiles,
                inchikey=node.inchikey,
                synthesis_step=new_step,
                metadata=node.metadata.copy(),
            )

    main_target = _rebuild(route.target)

    result: list[Route] = []

    # Only include main route if it still has reactions
    if main_target.synthesis_step is not None:
        main_route = Route(
            target=main_target,
            rank=route.rank,
            solvability=route.solvability.copy(),
            metadata=route.metadata.copy(),
        )
        result.append(main_route)

    # Add sub-routes (already filtered for having reactions)
    result.extend(sub_routes)

    return result
