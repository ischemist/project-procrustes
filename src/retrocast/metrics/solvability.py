from retrocast.models.chem import Route
from retrocast.typing import SmilesStr


def is_route_solved(route: Route, stock: set[SmilesStr]) -> bool:
    """
    Determines if a route is solvable given a set of stock compounds.

    A route is solved if ALL its leaf nodes (starting materials)
    are present in the stock.
    """
    return all(leaf.smiles in stock for leaf in route.leaves)
