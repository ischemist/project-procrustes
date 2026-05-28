from __future__ import annotations

from collections.abc import Sequence

from retrocast.v2.models.route import Molecule, ReactionView, Route, RoutePath


def iter_molecules(molecule: Molecule) -> Sequence[Molecule]:
    molecules = [molecule]
    if molecule.product_of is not None:
        for reactant in molecule.product_of.reactants:
            molecules.extend(iter_molecules(reactant))
    return molecules


def iter_reactions(route: Route) -> Sequence[ReactionView]:
    reactions = []

    def visit(path: RoutePath) -> None:
        molecule = route.molecule_at(path)
        reaction = molecule.produced_by()
        if reaction is None:
            return
        reactions.append(reaction)
        for index, _ in enumerate(reaction.value.reactants):
            visit(reaction.path.reactant(index))

    visit(RoutePath.target())
    return reactions


def leaf_molecules(route: Route) -> Sequence[Molecule]:
    return [molecule for molecule in iter_molecules(route.target) if molecule.product_of is None]
