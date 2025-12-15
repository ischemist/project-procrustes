from retrocast.chem import InchiKeyLevel
from retrocast.models.chem import Route


def find_acceptable_match(
    route: Route, acceptable_signatures: list[str], match_level: InchiKeyLevel = InchiKeyLevel.FULL
) -> int | None:
    """
    Finds the index of the first matching acceptable route.

    This function checks if the predicted route matches any of the acceptable routes
    by comparing topological signatures.

    Args:
        route: The predicted route to check
        acceptable_signatures: Pre-computed signatures of acceptable routes
        match_level: Level of InChI key matching specificity:
            - None or FULL: Exact matching (default)
            - NO_STEREO: Ignore stereochemistry
            - CONNECTIVITY: Match on molecular skeleton only

    Returns:
        Index of the first matching acceptable route, or None if no match

    Example:
        >>> acceptable_sigs = [r.get_signature() for r in target.acceptable_routes]
        >>> matched_idx = find_acceptable_match(predicted_route, acceptable_sigs)
        >>> if matched_idx is not None:
        ...     matched_route = target.acceptable_routes[matched_idx]
    """
    route_sig = route.get_signature(match_level=match_level)
    for idx, acceptable_sig in enumerate(acceptable_signatures):
        if route_sig == acceptable_sig:
            return idx
    return None
