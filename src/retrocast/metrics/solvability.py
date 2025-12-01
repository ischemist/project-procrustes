from retrocast.models.chem import Route
from retrocast.typing import InchiKeyStr


def is_route_solved(route: Route, stock: set[InchiKeyStr]) -> bool:
    """
    Determines if a route is solvable given a set of stock compounds.

    A route is solved if ALL its leaf nodes (starting materials)
    are present in the stock, based on InChI key matching.

    InChI-based matching is chemically correct and handles:
    - Tautomers (same molecule, different representations)
    - Stereoisomers (if not specified in InChI)
    - Canonical representation differences

    Args:
        route: The synthesis route to check
        stock: Set of InChI keys representing available stock molecules

    Returns:
        True if all starting materials are in stock, False otherwise
    """
    return all(leaf.inchikey in stock for leaf in route.leaves)
