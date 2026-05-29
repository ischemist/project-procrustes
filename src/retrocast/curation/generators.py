"""Route generation utilities for benchmark curation.

The main use case is expanding a ground-truth route into acceptable variants by
stopping synthesis at intermediates that are available in stock. This lets a
benchmark accept both "make the intermediate" and "buy the intermediate" forms
of the same route when the task stock makes both chemically reasonable.
"""

from __future__ import annotations

from itertools import combinations

from retrocast.curation.filtering import deduplicate_routes
from retrocast.models.route import Molecule, Route, RoutePath
from retrocast.typing import InChIKeyStr


def generate_pruned_routes(route: Route, stock: set[InChIKeyStr]) -> list[Route]:
    """Generate stock-solvable route variants by pruning at stock intermediates.

    A stock intermediate is a non-root, non-leaf molecule whose InChIKey appears
    in ``stock``. Each such molecule can become a leaf in an acceptable variant,
    cutting off the synthesis subtree that produced it.

    Prune choices are generated as antichains: no selected molecule may be an
    ancestor of another selected molecule. Without this filter, a linear route
    with stock intermediates ``target <- C <- B <- A`` would generate both
    ``{C}`` and ``{C, B}``, but pruning at ``C`` already removes ``B`` from the
    remaining route, so the two choices produce the same route. Independent
    branches still combine normally, so convergent routes get the expected
    multiplicative variants.

    Only stock-solvable variants are returned, meaning every leaf in the final
    route must be present in ``stock``. The original route is included through
    the empty antichain when it is stock-solvable. Leaf-only routes return an
    empty list because there is no reaction route to expand.

    Example:
        For ``D <- C <- B <- A`` where ``A``, ``B``, and ``C`` are in stock,
        this returns three routes: the original route, the route pruned at
        ``B``, and the route pruned at ``C``. It does not return a separate
        ``{B, C}`` variant because that is redundant with pruning at ``C``.
    """
    if route.target.product_of is None:
        return []

    stock_keys = {str(inchikey) for inchikey in stock}
    intermediate_paths = _stock_intermediate_paths(route.target, stock_keys)
    solvable_routes = []
    for prune_paths in _generate_antichains(intermediate_paths):
        pruned_route = Route(
            target=_rebuild_molecule(route.target, RoutePath.target(), prune_paths),
            annotations=route.annotations.copy(),
        )
        if all(str(leaf.value.inchikey) in stock_keys for leaf in pruned_route.iter_leaves()):
            solvable_routes.append(pruned_route)
    return deduplicate_routes(solvable_routes)


def _stock_intermediate_paths(molecule: Molecule, stock_keys: set[str]) -> list[RoutePath]:
    """Return paths to non-root, non-leaf stock molecules."""
    paths: list[RoutePath] = []

    def traverse(current: Molecule, path: RoutePath) -> None:
        reaction = current.product_of
        if reaction is None:
            return

        if path.indices and str(current.inchikey) in stock_keys:
            paths.append(path)

        for index, reactant in enumerate(reaction.reactants):
            traverse(reactant, path.produced_by().reactant(index))

    traverse(molecule, RoutePath.target())
    return paths


def _generate_antichains(paths: list[RoutePath]) -> list[frozenset[RoutePath]]:
    """Return all path subsets where no selected path contains another."""
    antichains: list[frozenset[RoutePath]] = [frozenset()]
    for size in range(1, len(paths) + 1):
        for candidate in combinations(paths, size):
            if _is_antichain(candidate):
                antichains.append(frozenset(candidate))
    return antichains


def _is_antichain(paths: tuple[RoutePath, ...]) -> bool:
    return all(not _is_ancestor(left, right) for left, right in combinations(paths, 2))


def _is_ancestor(left: RoutePath, right: RoutePath) -> bool:
    shorter, longer = (
        (left.indices, right.indices) if len(left.indices) < len(right.indices) else (right.indices, left.indices)
    )
    return shorter == longer[: len(shorter)]


def _rebuild_molecule(molecule: Molecule, path: RoutePath, prune_paths: frozenset[RoutePath]) -> Molecule:
    if path in prune_paths or molecule.product_of is None:
        return molecule.model_copy(update={"product_of": None}, deep=True)

    reaction = molecule.product_of
    reactants = [
        _rebuild_molecule(reactant, path.produced_by().reactant(index), prune_paths)
        for index, reactant in enumerate(reaction.reactants)
    ]
    return molecule.model_copy(
        update={"product_of": reaction.model_copy(update={"reactants": reactants}, deep=True)},
        deep=True,
    )
