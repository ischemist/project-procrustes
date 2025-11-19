from retrocast.models.chem import Route


def is_exact_match(route: Route, gt_signature: str) -> bool:
    """
    Checks if a predicted route is topologically identical to the ground truth.

    Args:
        route: The predicted route.
        gt_signature: The pre-computed signature hash of the ground truth route.
    """
    return route.get_signature() == gt_signature
