from retrocast.chem import InchiKeyLevel, reduce_inchikey
from retrocast.models.chem import Route
from retrocast.typing import InchiKeyStr


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
