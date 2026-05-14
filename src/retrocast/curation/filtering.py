import logging
from collections.abc import Callable, Hashable
from copy import deepcopy
from typing import Literal

from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route

logger = logging.getLogger(__name__)

RouteType = Literal["linear", "convergent"]


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
                metadata=deepcopy(node.metadata),
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
                            metadata=deepcopy(route.metadata),
                        )
                        sub_routes.append(new_route)

            # Return this node as a leaf (no synthesis_step)
            return Molecule(
                smiles=node.smiles,
                inchikey=node.inchikey,
                metadata=deepcopy(node.metadata),
            )
        else:
            # Keep this reaction, but recursively check reactants
            new_reactants = [_rebuild(r) for r in rxn.reactants]
            new_step = ReactionStep(
                reactants=new_reactants,
                mapped_smiles=rxn.mapped_smiles,
                template=rxn.template,
                reagents=list(rxn.reagents) if rxn.reagents is not None else None,
                solvents=list(rxn.solvents) if rxn.solvents is not None else None,
                metadata=deepcopy(rxn.metadata),
            )
            return Molecule(
                smiles=node.smiles,
                inchikey=node.inchikey,
                synthesis_step=new_step,
                metadata=deepcopy(node.metadata),
            )

    main_target = _rebuild(route.target)

    result: list[Route] = []

    # Only include main route if it still has reactions
    if main_target.synthesis_step is not None:
        main_route = Route(
            target=main_target,
            rank=route.rank,
            metadata=deepcopy(route.metadata),
        )
        result.append(main_route)

    # Add sub-routes (already filtered for having reactions)
    result.extend(sub_routes)

    return result


def deduplicate_routes(
    routes: list[Route],
    *,
    key: Callable[[Route], Hashable] | None = None,
) -> list[Route]:
    """
    Filters a list of Route objects, returning only the unique routes.

    Args:
        routes: Routes to deduplicate.
        key: Optional identity function. Defaults to Route.get_structural_signature().
    """
    route_key = key or (lambda route: route.get_structural_signature())
    seen_keys: set[Hashable] = set()
    unique_routes = []

    logger.debug(f"Deduplicating {len(routes)} routes...")

    for route in routes:
        route_identity = route_key(route)

        if route_identity not in seen_keys:
            seen_keys.add(route_identity)
            unique_routes.append(route)

    num_removed = len(routes) - len(unique_routes)
    if num_removed > 0:
        logger.debug(f"Removed {num_removed} duplicate routes.")

    return unique_routes


def filter_by_route_type(benchmark: BenchmarkSet, route_type: RouteType) -> list[BenchmarkTarget]:
    """
    Extracts targets based on their primary acceptable route's synthesis topology.

    Uses the first acceptable route (primary route) to determine convergence.
    Targets without acceptable routes are excluded.

    Args:
        benchmark: The benchmark set to filter
        route_type: "convergent" or "linear"

    Returns:
        List of targets matching the specified route type
    """
    if route_type == "convergent":
        return [t for t in benchmark.targets.values() if t.is_convergent is True]
    elif route_type == "linear":
        return [t for t in benchmark.targets.values() if t.is_convergent is False]
    else:
        raise ValueError(f"Unknown route type: {route_type}")


def clean_and_prioritize_pools(
    primary: list[BenchmarkTarget], secondary: list[BenchmarkTarget]
) -> tuple[list[BenchmarkTarget], list[BenchmarkTarget]]:
    """
    Cleans two pools of targets based on conflicts, but keeps them separate.

    Removes duplicates based on primary acceptable route signatures and
    identifies/removes targets with ambiguous SMILES across pools.

    Returns:
        (cleaned_primary, cleaned_secondary)
    """
    logger.info(f"Cleaning pools: Primary ({len(primary)}) + Secondary ({len(secondary)})")

    # 1. Filter Secondary by Route Signature (remove duplicates based on primary route)
    primary_sigs = {t.primary_route.get_structural_signature() for t in primary if t.primary_route}

    secondary_unique_routes = []
    for t in secondary:
        if t.primary_route and t.primary_route.get_structural_signature() in primary_sigs:
            continue
        secondary_unique_routes.append(t)

    # 2. Identify Ambiguous SMILES
    primary_smiles = {t.smiles for t in primary}
    secondary_smiles = {t.smiles for t in secondary_unique_routes}
    ambiguous_smiles = primary_smiles.intersection(secondary_smiles)

    if ambiguous_smiles:
        logger.warning(f"  Removing {len(ambiguous_smiles)} ambiguous targets from BOTH pools.")

    # 3. Construct Final Lists
    clean_primary = [t for t in primary if t.smiles not in ambiguous_smiles]
    clean_secondary = [t for t in secondary_unique_routes if t.smiles not in ambiguous_smiles]

    logger.info(f"Cleaned sizes: Primary {len(clean_primary)}, Secondary {len(clean_secondary)}")

    return clean_primary, clean_secondary
