"""Filtering and route surgery helpers for benchmark curation.

These utilities operate on already-canonical v2 ``Route`` and ``Benchmark``
objects. They are intentionally topology-oriented: the code compares route and
reaction signatures, removes reaction-overlap fragments, classifies route shape,
and keeps curation pools from leaking duplicate or ambiguous targets.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Literal

from retrocast.models.route import Molecule, Route, RoutePath
from retrocast.models.task import Benchmark, Target

RouteType = Literal["linear", "convergent"]


def excise_reactions_from_route(route: Route, exclude: set[str]) -> list[Route]:
    """Cut matching reactions out of ``route`` and return non-empty fragments.

    ``exclude`` contains v2 reaction signatures, i.e. values produced by
    ``route.reaction_at(path).signature()``. When a matching reaction is found,
    its product molecule is retained but converted to a leaf in the rebuilt
    parent route. Any reactant branch below the cut that still contains at least
    one reaction becomes its own returned subroute.

    The returned routes are ordered as the rebuilt main route first, if it still
    contains any reactions, followed by subroutes discovered during depth-first
    traversal. Leaf-only fragments are intentionally dropped because they are not
    useful training routes.

    Example:
        Given ``R <- A1 <- A2 <- I1 <- I2 <- A3`` and excluding the reaction
        that produces ``I1`` from ``I2``, the result is a main route ending at
        ``I1`` as a leaf plus a subroute rooted at ``I2``.
    """
    if route.target.product_of is None:
        return []

    subroutes = []

    def rebuild(molecule: Molecule, path: RoutePath) -> Molecule:
        reaction = molecule.product_of
        if reaction is None:
            return molecule.model_copy(deep=True)

        reaction_view = route.reaction_at(path.produced_by())
        if reaction_view.signature() in exclude:
            for index, reactant in enumerate(reaction.reactants):
                if reactant.product_of is None:
                    continue
                rebuilt = rebuild(reactant, path.produced_by().reactant(index))
                if rebuilt.product_of is not None:
                    subroutes.append(Route(target=rebuilt, annotations=route.annotations.copy()))
            return molecule.model_copy(update={"product_of": None}, deep=True)

        reactants = [
            rebuild(reactant, path.produced_by().reactant(index)) for index, reactant in enumerate(reaction.reactants)
        ]
        return molecule.model_copy(
            update={"product_of": reaction.model_copy(update={"reactants": reactants}, deep=True)},
            deep=True,
        )

    rebuilt_target = rebuild(route.target, RoutePath.target())
    routes = []
    if rebuilt_target.product_of is not None:
        routes.append(Route(target=rebuilt_target, annotations=route.annotations.copy()))
    routes.extend(subroutes)
    return routes


def deduplicate_routes(
    routes: list[Route],
    *,
    key: Callable[[Route], Hashable] | None = None,
) -> list[Route]:
    """Return routes with duplicate identities removed, preserving first occurrence.

    By default, identity is the full v2 route signature. Callers may pass
    ``key`` to deduplicate by a broader or narrower identity, such as depth,
    target key, or a depth-limited signature.
    """
    route_key = key or (lambda route: route.signature())
    seen = set()
    output = []
    for route in routes:
        identity = route_key(route)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(route)
    return output


def filter_by_route_type(benchmark: Benchmark, route_type: RouteType) -> list[Target]:
    """Return benchmark targets whose primary acceptable route has the requested topology.

    Route type is derived from the first acceptable route for each target. Targets
    without acceptable routes are excluded because there is no topology to
    classify. ``linear`` means no reaction in the route has more than one
    synthesized reactant branch; ``convergent`` means at least one reaction does.
    """
    if route_type not in ("linear", "convergent"):
        raise ValueError(f"Unknown route type: {route_type}")
    want_convergent = route_type == "convergent"
    return [
        target
        for target in benchmark.targets.values()
        if target.acceptable_routes and target.acceptable_routes[0].is_convergent() is want_convergent
    ]


def clean_and_prioritize_pools(primary: list[Target], secondary: list[Target]) -> tuple[list[Target], list[Target]]:
    """Remove route duplicates and ambiguous targets across prioritized pools.

    The primary pool wins route-signature conflicts: secondary targets whose
    first acceptable route duplicates a primary target's first acceptable route
    are removed. After that, any target SMILES that still appears in both pools
    is considered ambiguous and removed from both pools.

    Targets without acceptable routes are kept unless they collide by SMILES,
    because there is no route signature to compare.
    """
    primary_signatures = {target.acceptable_routes[0].signature() for target in primary if target.acceptable_routes}
    secondary_without_route_dupes = [
        target
        for target in secondary
        if not target.acceptable_routes or target.acceptable_routes[0].signature() not in primary_signatures
    ]
    ambiguous_smiles = {target.smiles for target in primary} & {
        target.smiles for target in secondary_without_route_dupes
    }
    return (
        [target for target in primary if target.smiles not in ambiguous_smiles],
        [target for target in secondary_without_route_dupes if target.smiles not in ambiguous_smiles],
    )


def route_is_convergent(route: Route) -> bool:
    """Return whether any reaction in ``route`` combines multiple synthesized branches.

    A reaction with one synthesized reactant and any number of leaf reactants is
    still considered linear. A route is convergent only when a single reaction
    joins two or more reactants that themselves have producing reactions.

    This curation-level wrapper is kept for existing callers; new code can call
    ``Route.is_convergent()`` directly.
    """
    return route.is_convergent()
