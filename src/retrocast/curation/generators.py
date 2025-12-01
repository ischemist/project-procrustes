"""
Route generation utilities for benchmark curation.

This module provides functions for generating alternative acceptable routes
from ground truth routes, such as by pruning at stock-available intermediates.
"""

from __future__ import annotations

from itertools import combinations

from retrocast.metrics.solvability import is_route_solved
from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr


def get_stock_intermediates(route: Route, stock: set[InchiKeyStr]) -> set[Molecule]:
    """
    Identify all intermediate molecules in a route that are available in stock.

    An intermediate is defined as a non-leaf, non-target molecule in the route tree.
    These represent synthesized molecules that could alternatively be purchased
    if available in stock.

    Args:
        route: The synthesis route to analyze
        stock: Set of InChIKeys representing available stock molecules

    Returns:
        Set of Molecule objects representing intermediates found in stock.
        Returns empty set if no intermediates are in stock.

    Examples:
        >>> # Route: D <- C <- B <- A (where A is leaf, D is target)
        >>> # If B and C are in stock, returns {B, C}
        >>> stock_ints = get_stock_intermediates(route, stock)
    """
    intermediates: set[Molecule] = set()
    target_inchikey = route.target.inchikey

    def _collect_intermediates(mol: Molecule) -> None:
        """Recursively collect intermediate molecules."""
        if mol.is_leaf:
            return

        # Non-leaf molecule: check if it's an intermediate (not the target)
        if mol.inchikey != target_inchikey and mol.inchikey in stock:
            intermediates.add(mol)

        # Recursively process reactants
        if mol.synthesis_step:
            for reactant in mol.synthesis_step.reactants:
                _collect_intermediates(reactant)

    _collect_intermediates(route.target)
    return intermediates


def generate_pruned_routes(route: Route, stock: set[InchiKeyStr]) -> list[Route]:
    """
    Generate all possible route variants by pruning at stock-available intermediates.

    Given a ground truth route where intermediates may be available in stock,
    this generates alternative acceptable routes by treating those intermediates
    as starting materials (leaves) instead of synthesized products.

    Uses antichain filtering to avoid generating redundant routes. In a linear route
    where intermediate B is downstream of A, pruning at A makes B unreachable, so
    the combination {A, B} produces the same route as {A} alone. Only generates
    antichains (sets where no element is an ancestor of another).

    Args:
        route: The original synthesis route
        stock: Set of InChIKeys representing available stock molecules

    Returns:
        List of Route objects, each representing a valid pruned variant.
        Returns empty list if the route is a leaf or has no reactions.
        Original route is always included if it's solvable with stock.

    Algorithm:
        1. Find all intermediates that are in stock
        2. Build ancestor map (which intermediates are ancestors of which)
        3. Generate only valid antichains (no ancestor-descendant pairs)
        4. For each antichain, create a pruned route
        5. Filter to only routes that are solvable (all leaves in stock)

    Examples:
        >>> # Route: D <- C <- B <- A, where B and C are in stock
        >>> # Generates 3 routes (not 4):
        >>> # 1. D <- C <- B <- A (original, no pruning)
        >>> # 2. D <- C <- B (prune at B only)
        >>> # 3. D <- C (prune at C only)
        >>> # Note: {B, C} is invalid because C is ancestor of B
        >>> routes = generate_pruned_routes(route, stock)

    Performance:
        - O(k * n) for ancestor map, O(2^k * k) for antichain filtering
        - Linear routes (90%): generates ~depth routes instead of 2^depth
        - Convergent routes: intermediates in different branches multiply independently
    """
    # Handle edge case: leaf-only route has no reactions to prune
    if route.target.is_leaf:
        return []

    # Find all stock-available intermediates
    stock_intermediates = get_stock_intermediates(route, stock)

    if not stock_intermediates:
        # No intermediates in stock: only generate routes if original is solvable
        if is_route_solved(route, stock):
            return [route]
        else:
            return []

    # Build ancestor map for pre-filtering
    ancestor_map = _build_ancestor_map(route, stock)

    # Generate only valid antichains (no ancestor-descendant pairs in same set)
    valid_prune_combinations = _generate_antichains(stock_intermediates, ancestor_map)

    pruned_routes = []

    for prune_combination in valid_prune_combinations:
        # Create set of InChIKeys to prune at
        prune_at = frozenset(mol.inchikey for mol in prune_combination)

        # Generate pruned route
        pruned_route = _create_pruned_route(route, prune_at)

        # Only include if route is solvable with the stock
        if is_route_solved(pruned_route, stock):
            pruned_routes.append(pruned_route)

    return pruned_routes


def _create_pruned_route(route: Route, prune_at: frozenset[InchiKeyStr]) -> Route:
    """
    Create a new route by pruning at specified intermediate molecules.

    Molecules in prune_at become leaves in the new route, cutting off their
    synthesis subtrees.

    Args:
        route: Original route
        prune_at: Set of InChIKeys where pruning should occur

    Returns:
        New Route object with pruned structure
    """

    def _rebuild_molecule(mol: Molecule) -> Molecule:
        """Recursively rebuild molecule tree with pruning."""
        # If this molecule should be pruned, make it a leaf
        if mol.inchikey in prune_at:
            return Molecule(
                smiles=mol.smiles,
                inchikey=mol.inchikey,
                synthesis_step=None,  # Make it a leaf
                metadata=mol.metadata.copy(),
            )

        # If it's already a leaf, keep it as is
        if mol.is_leaf:
            return Molecule(
                smiles=mol.smiles,
                inchikey=mol.inchikey,
                synthesis_step=None,
                metadata=mol.metadata.copy(),
            )

        # Otherwise, recursively rebuild the synthesis step
        assert mol.synthesis_step is not None
        new_reactants = [_rebuild_molecule(r) for r in mol.synthesis_step.reactants]

        new_step = ReactionStep(
            reactants=new_reactants,
            mapped_smiles=mol.synthesis_step.mapped_smiles,
            template=mol.synthesis_step.template,
            reagents=mol.synthesis_step.reagents,
            solvents=mol.synthesis_step.solvents,
            metadata=mol.synthesis_step.metadata.copy(),
        )

        return Molecule(
            smiles=mol.smiles,
            inchikey=mol.inchikey,
            synthesis_step=new_step,
            metadata=mol.metadata.copy(),
        )

    # Rebuild the target molecule with pruning
    new_target = _rebuild_molecule(route.target)

    # Create new route preserving metadata and rank
    return Route(
        target=new_target,
        rank=route.rank,
        metadata=route.metadata.copy(),
    )


def _build_ancestor_map(route: Route, stock: set[InchiKeyStr]) -> dict[InchiKeyStr, set[InchiKeyStr]]:
    """
    Build a map of which stock intermediates are ancestors of which.

    For each stock intermediate, tracks the InChIKeys of all its ancestors
    that are also stock intermediates. This enables filtering out redundant
    pruning combinations.

    Args:
        route: The synthesis route to analyze
        stock: Set of InChIKeys representing available stock molecules

    Returns:
        Dictionary mapping each stock intermediate's InChIKey to a set of
        ancestor InChIKeys that are also stock intermediates.

    Example:
        Route: D <- C <- B <- A (where B, C are stock intermediates)
        Returns: {
            B_inchikey: {},           # B has no stock-intermediate ancestors
            C_inchikey: {B_inchikey}  # C's ancestor B is also a stock intermediate
        }
    """
    target_inchikey = route.target.inchikey

    def _traverse(mol: Molecule, ancestors: set[InchiKeyStr]) -> dict[InchiKeyStr, set[InchiKeyStr]]:
        """Recursively traverse route, tracking stock intermediate ancestors."""
        result = {}

        # Is this molecule a stock intermediate?
        is_stock_intermediate = not mol.is_leaf and mol.inchikey != target_inchikey and mol.inchikey in stock

        if is_stock_intermediate:
            # Record which ancestors (that are also stock intermediates) this node has
            result[mol.inchikey] = ancestors.copy()

        # Recurse into children
        if mol.synthesis_step:
            # If this is a stock intermediate, add it to ancestors for children
            new_ancestors = ancestors | ({mol.inchikey} if is_stock_intermediate else set())
            for reactant in mol.synthesis_step.reactants:
                child_result = _traverse(reactant, new_ancestors)
                result.update(child_result)

        return result

    return _traverse(route.target, set())


def _generate_antichains(
    elements: set[Molecule], ancestor_map: dict[InchiKeyStr, set[InchiKeyStr]]
) -> list[frozenset[Molecule]]:
    """
    Generate all antichains from a set of elements.

    An antichain is a subset where no element is an ancestor of another.
    This avoids redundant pruning combinations where pruning at ancestor A
    makes descendant B unreachable.

    Args:
        elements: Set of Molecule objects (stock intermediates)
        ancestor_map: Map of {inchikey: set of ancestor inchikeys}

    Returns:
        List of frozensets, each representing a valid pruning combination.
        Includes empty frozenset (no pruning).

    Example:
        Elements: {B, C} where C is ancestor of B
        Returns: [{}, {B}, {C}]  # Not {B, C} since C is ancestor of B
    """
    antichains: list[frozenset[Molecule]] = [frozenset()]  # Empty set is always valid

    element_list = list(elements)

    # Generate all combinations of all sizes
    for r in range(1, len(element_list) + 1):
        for combo in combinations(element_list, r):
            combo_inchikeys = {mol.inchikey for mol in combo}

            # Check if this is an antichain (no element is ancestor of another)
            is_antichain = True
            for mol in combo:
                # If any of mol's ancestors are also in this combo, it's not an antichain
                mol_ancestors = ancestor_map.get(mol.inchikey, set())
                if mol_ancestors & combo_inchikeys:
                    is_antichain = False
                    break

            if is_antichain:
                antichains.append(frozenset(combo))

    return antichains
