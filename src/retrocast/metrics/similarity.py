from retrocast.models.chem import Route


def find_acceptable_match(route: Route, acceptable_signatures: list[str]) -> int | None:
    """
    Finds the index of the first matching acceptable route.

    This function checks if the predicted route matches any of the acceptable routes
    by comparing topological signatures.

    Args:
        route: The predicted route to check
        acceptable_signatures: Pre-computed signatures of acceptable routes

    Returns:
        Index of the first matching acceptable route, or None if no match

    Example:
        >>> acceptable_sigs = [r.get_signature() for r in target.acceptable_routes]
        >>> matched_idx = find_acceptable_match(predicted_route, acceptable_sigs)
        >>> if matched_idx is not None:
        ...     matched_route = target.acceptable_routes[matched_idx]
    """
    route_sig = route.get_signature()
    for idx, acceptable_sig in enumerate(acceptable_signatures):
        if route_sig == acceptable_sig:
            return idx
    return None
