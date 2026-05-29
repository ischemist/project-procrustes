from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.curation import (
    clean_and_prioritize_pools,
    deduplicate_routes,
    excise_reactions_from_route,
    filter_by_route_type,
    route_is_convergent,
)
from retrocast.models import Benchmark, Molecule, Reaction, Route, RoutePath, Target
from retrocast.typing import InChIKeyStr, SmilesStr


@st.composite
def route_shape(draw: st.DrawFn) -> tuple[object, ...]:
    subtree = st.recursive(
        st.none(),
        lambda inner: st.lists(inner, min_size=1, max_size=3).map(tuple),
        max_leaves=8,
    )
    return tuple(draw(st.lists(subtree, min_size=1, max_size=3)))


def test_deduplicate_routes_uses_route_signature() -> None:
    route = linear_route()

    assert deduplicate_routes([route, route.model_copy(deep=True)]) == [route]


def test_filter_by_route_type_uses_primary_acceptable_route() -> None:
    linear_target = target("linear", "CCO", linear_route())
    convergent_target = target("convergent", "CCN", convergent_route())
    benchmark = Benchmark(
        name="routes",
        targets={linear_target.id: linear_target, convergent_target.id: convergent_target},
    )

    assert filter_by_route_type(benchmark, "linear") == [linear_target]
    assert filter_by_route_type(benchmark, "convergent") == [convergent_target]
    assert route_is_convergent(linear_route()) is False
    assert route_is_convergent(convergent_route()) is True
    assert linear_route().is_convergent() is False
    assert convergent_route().is_convergent() is True


def test_clean_and_prioritize_pools_removes_secondary_route_duplicates_and_ambiguous_smiles() -> None:
    primary = [target("a", "CCO", linear_route()), target("b", "CCC", route_for("CCC"))]
    secondary = [
        target("route-dupe", "CCO", linear_route()),
        target("ambiguous", "CCC", route_for("CCN")),
        target("kept", "CCN", route_for("CCN")),
    ]

    clean_primary, clean_secondary = clean_and_prioritize_pools(primary, secondary)

    assert [item.id for item in clean_primary] == ["a"]
    assert [item.id for item in clean_secondary] == ["kept"]


def test_excise_reactions_from_route_returns_remaining_subroutes() -> None:
    route = linear_route()
    root_signature = route.reaction_at("rc:r:/").signature()

    routes = excise_reactions_from_route(route, {root_signature})

    assert len(routes) == 1
    assert routes[0].target.smiles == SmilesStr(canonicalize_smiles("CC"))


def test_excise_middle_reaction_splits_linear_route() -> None:
    route = three_step_linear_route()
    middle_signature = route.reaction_at("rc:r:/0").signature()

    fragments = excise_reactions_from_route(route, {middle_signature})

    assert len(fragments) == 2
    assert fragments[0].target.smiles == SmilesStr(canonicalize_smiles("CCCC"))
    assert route_leaf_smiles(fragments[0]) == {canonicalize_smiles("CCC")}
    assert fragments[1].target.smiles == SmilesStr(canonicalize_smiles("CC"))
    assert route_reaction_signatures(fragments[0]).isdisjoint({middle_signature})
    assert route_reaction_signatures(fragments[1]).isdisjoint({middle_signature})


def test_excise_top_convergent_reaction_keeps_only_live_subroutes() -> None:
    live_left = molecule("CC", product_of=Reaction(reactants=[molecule("C")]))
    live_right = molecule("CN", product_of=Reaction(reactants=[molecule("N")]))
    leaf = molecule("O")
    route = Route(target=molecule("CCNO", product_of=Reaction(reactants=[live_left, leaf, live_right])))

    fragments = excise_reactions_from_route(route, {route.reaction_at("rc:r:/").signature()})

    assert [fragment.target.smiles for fragment in fragments] == [
        SmilesStr(canonicalize_smiles("CC")),
        SmilesStr(canonicalize_smiles("CN")),
    ]


@given(shape=route_shape(), data=st.data())
@settings(max_examples=75)
def test_excise_partitions_remaining_reactions(shape: tuple[object, ...], data: st.DataObject) -> None:
    route = route_from_shape(shape)
    signatures = tuple(sorted(route_reaction_signatures(route)))
    exclude = data.draw(st.sets(st.sampled_from(signatures), max_size=len(signatures)))

    fragments = excise_reactions_from_route(route, exclude)

    assert fragment_component_map(fragments) == expected_component_map(route, exclude)
    for fragment in fragments:
        assert fragment.depth() > 0
        assert route_reaction_signatures(fragment).isdisjoint(exclude)


def linear_route() -> Route:
    cc = molecule("CC", product_of=Reaction(reactants=[molecule("C")]))
    target_molecule = molecule("CCO", product_of=Reaction(reactants=[molecule("C"), cc]))
    return Route(target=target_molecule)


def three_step_linear_route() -> Route:
    leaf = molecule("C")
    bottom = molecule("CC", product_of=Reaction(reactants=[leaf]))
    middle = molecule("CCC", product_of=Reaction(reactants=[bottom]))
    return Route(target=molecule("CCCC", product_of=Reaction(reactants=[middle])))


def convergent_route() -> Route:
    left = molecule("CC", product_of=Reaction(reactants=[molecule("C")]))
    right = molecule("CN", product_of=Reaction(reactants=[molecule("N")]))
    return Route(target=molecule("CCN", product_of=Reaction(reactants=[left, right])))


def route_for(smiles: str) -> Route:
    return Route(target=molecule(smiles, product_of=Reaction(reactants=[molecule("C")])))


def target(target_id: str, smiles: str, route: Route) -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(
        id=target_id,
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        acceptable_routes=[route],
    )


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical)), product_of=product_of)


def route_from_shape(shape: tuple[object, ...]) -> Route:
    node_index = 0

    def build(node_shape: object) -> Molecule:
        nonlocal node_index
        node_index += 1
        smiles = "C" * node_index
        if node_shape is None:
            return molecule(smiles)
        assert isinstance(node_shape, tuple)
        return molecule(smiles, product_of=Reaction(reactants=[build(child) for child in node_shape]))

    return Route(target=build(shape))


def route_reaction_signatures(route: Route) -> set[str]:
    signatures = set()
    stack = [RoutePath.root_reaction()]
    while stack:
        path = stack.pop()
        try:
            reaction = route.reaction_at(path)
        except KeyError:
            continue
        signatures.add(reaction.signature())
        stack.extend(
            reactant.path.produced_by() for reactant in reaction.reactants() if reactant.value.product_of is not None
        )
    return signatures


def fragment_component_map(fragments: list[Route]) -> dict[str, set[str]]:
    return {str(fragment.target.inchikey): route_reaction_signatures(fragment) for fragment in fragments}


def expected_component_map(route: Route, exclude: set[str]) -> dict[str, set[str]]:
    components: dict[str, set[str]] = {}

    def visit(molecule: Molecule, path: RoutePath, current_root: str | None) -> None:
        reaction = molecule.product_of
        if reaction is None:
            return

        signature = route.reaction_at(path.produced_by()).signature()
        if signature in exclude:
            for index, reactant in enumerate(reaction.reactants):
                visit(reactant, path.produced_by().reactant(index), None)
            return

        if current_root is None:
            current_root = str(molecule.inchikey)
            components[current_root] = set()
        components[current_root].add(signature)
        for index, reactant in enumerate(reaction.reactants):
            visit(reactant, path.produced_by().reactant(index), current_root)

    visit(route.target, RoutePath.target(), None)
    return components


def route_leaf_smiles(route: Route) -> set[str]:
    return {str(leaf.value.smiles) for leaf in route.iter_leaves()}
