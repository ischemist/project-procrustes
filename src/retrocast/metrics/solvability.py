from __future__ import annotations

from retrocast.chem import InchiKeyLevel, reduce_inchikey
from retrocast.exceptions import InputError, UnsupportedValidityTierError
from retrocast.models.chem import Route, RouteReaction
from retrocast.models.validity import IMPLEMENTED_VALIDITY_TIERS, SUPPORTED_VALIDITY_TIERS, ValidityTier
from retrocast.typing import InchiKeyStr

TIER0_EMPTY_REACTANTS = "tier0.empty_reactants"


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


def _ensure_implemented_tier(tier: ValidityTier) -> None:
    if tier not in SUPPORTED_VALIDITY_TIERS:
        raise InputError(
            f"Unknown validity tier: {tier}.",
            code="validity.unknown_tier",
            context={"tier": tier, "supported_tiers": sorted(SUPPORTED_VALIDITY_TIERS)},
        )
    if tier in IMPLEMENTED_VALIDITY_TIERS:
        return
    raise UnsupportedValidityTierError(
        f"Tier-{tier} validity is not implemented.",
        context={"tier": tier, "implemented_tiers": sorted(IMPLEMENTED_VALIDITY_TIERS)},
    )


def get_reaction_tier_failure_codes(reaction: RouteReaction, tier: ValidityTier) -> list[str]:
    """Return stable failure codes for one route reaction at the requested validity tier."""
    _ensure_implemented_tier(tier)

    codes: list[str] = []
    if not reaction.step.reactants:
        codes.append(TIER0_EMPTY_REACTANTS)
    return sorted(set(codes))


def get_route_tier_failure_codes(route: Route, tier: ValidityTier) -> list[str]:
    """Return stable failure codes for a route at the requested validity tier."""
    _ensure_implemented_tier(tier)

    reactions = list(route.iter_reactions())
    if not reactions:
        return []

    codes: list[str] = []
    for reaction in reactions:
        codes.extend(get_reaction_tier_failure_codes(reaction, tier))
    return sorted(set(codes))


def is_route_tier_valid(route: Route, tier: ValidityTier) -> bool:
    """Return whether a route passes the requested validity tier."""
    return not get_route_tier_failure_codes(route, tier)
