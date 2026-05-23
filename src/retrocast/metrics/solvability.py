from __future__ import annotations

from collections.abc import Iterator

from retrocast.chem import InchiKeyLevel, canonicalize_smiles, reduce_inchikey
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Molecule, Route, RouteReaction
from retrocast.typing import InchiKeyStr

TIER0_EMPTY_REACTANTS = "tier0.empty_reactants"
TIER0_INVALID_PRODUCT_SMILES = "tier0.invalid_product_smiles"
TIER0_INVALID_REACTANT_SMILES = "tier0.invalid_reactant_smiles"
TIER0_INVALID_TARGET_SMILES = "tier0.invalid_target_smiles"


def is_route_solved(
    route: Route,
    stock: set[InchiKeyStr],
    match_level: InchiKeyLevel = InchiKeyLevel.FULL,
) -> bool:
    """
    Determines if a route is solvable given a set of stock compounds.

    A route is solved if ALL its leaf nodes (starting materials)
    are present in the stock, based on InChI key matching.

    InChI-based matching is chemically correct and handles:
    - Tautomers (same molecule, different representations)
    - Stereoisomers (when using NO_STEREO or CONNECTIVITY level)
    - Canonical representation differences

    Args:
        route: The synthesis route to check
        stock: Set of InChI keys representing available stock molecules
        match_level: Level of InChI key matching specificity:
            - None or FULL: Exact matching (default)
            - NO_STEREO: Ignore stereochemistry
            - CONNECTIVITY: Match on molecular skeleton only

    Returns:
        True if all starting materials are in stock, False otherwise
    """
    if match_level == InchiKeyLevel.FULL:
        return all(leaf.inchikey in stock for leaf in route.leaves)
    return all(reduce_inchikey(leaf.inchikey, match_level) in stock for leaf in route.leaves)


def iter_route_molecules(route: Route) -> Iterator[Molecule]:
    """Yield unique molecule nodes in deterministic root-first depth-first order."""
    seen: set[int] = set()

    def _visit(node: Molecule) -> Iterator[Molecule]:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        yield node
        if node.synthesis_step is None:
            return
        for reactant in node.synthesis_step.reactants:
            yield from _visit(reactant)

    yield from _visit(route.target)


def is_molecule_tier_0_valid(molecule: Molecule) -> bool:
    """Return whether a molecule has a syntactically valid molecular graph."""
    try:
        canonicalize_smiles(str(molecule.smiles))
    except RetroCastException:
        return False
    return True


def get_reaction_tier_0_failure_codes(reaction: RouteReaction) -> list[str]:
    """Return tier-0 syntax failure codes for one route reaction."""
    codes: list[str] = []
    if not reaction.step.reactants:
        codes.append(TIER0_EMPTY_REACTANTS)
    if not is_molecule_tier_0_valid(reaction.product):
        codes.append(TIER0_INVALID_PRODUCT_SMILES)
    if any(not is_molecule_tier_0_valid(reactant) for reactant in reaction.step.reactants):
        codes.append(TIER0_INVALID_REACTANT_SMILES)
    return sorted(set(codes))


def get_route_tier_0_failure_codes(route: Route) -> list[str]:
    """Return route-level tier-0 syntax failure codes."""
    reactions = route.get_reactions()
    if not reactions:
        if is_molecule_tier_0_valid(route.target):
            return []
        return [TIER0_INVALID_TARGET_SMILES]

    codes: list[str] = []
    for reaction in reactions:
        codes.extend(get_reaction_tier_0_failure_codes(reaction))
    return sorted(set(codes))


def is_route_tier_0_valid(route: Route) -> bool:
    """Return whether every molecule/reaction record in the route is syntactically valid."""
    return not get_route_tier_0_failure_codes(route)
