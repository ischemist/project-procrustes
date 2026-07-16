from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.curation.generators import generate_pruned_routes
from retrocast.models import Molecule, Reaction, Route
from retrocast.typing import InChIKeyStr, SmilesStr


@pytest.mark.unit
def test_no_stock_intermediates_returns_original_when_original_is_solvable() -> None:
    route = linear_route(depth=3)
    stock = {inchikey("C")}

    routes = generate_pruned_routes(route, stock)

    assert [candidate.signature() for candidate in routes] == [route.signature()]
    assert [candidate.depth() for candidate in routes] == [3]


@pytest.mark.unit
def test_chained_stock_intermediates_generate_only_singleton_antichains() -> None:
    route = linear_route(depth=4)
    stock = {inchikey("C"), inchikey("CC"), inchikey("CCC")}

    routes = generate_pruned_routes(route, stock)

    assert len(routes) == 3
    assert sorted(candidate.depth() for candidate in routes) == [2, 3, 4]
    assert route.signature() in {candidate.signature() for candidate in routes}


@pytest.mark.unit
def test_convergent_route_combines_independent_branch_prunes() -> None:
    route = convergent_route()
    stock = {inchikey("C"), inchikey("O"), inchikey("CC"), inchikey("CCC"), inchikey("CO")}

    routes = generate_pruned_routes(route, stock)

    assert len(routes) == 6
    assert len({candidate.signature() for candidate in routes}) == len(routes)
    assert any(route_leaf_smiles(candidate) == {"CCC", "CO"} for candidate in routes)
    assert all(route_leaves_are_in_stock(candidate, stock) for candidate in routes)


@pytest.mark.unit
def test_stock_with_only_intermediate_excludes_unsolved_original_route() -> None:
    route = linear_route(depth=3)
    stock = {inchikey("CC")}

    routes = generate_pruned_routes(route, stock)

    assert len(routes) == 1
    assert routes[0].depth() == 2
    assert route_leaf_smiles(routes[0]) == {"CC"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "expected_depths"),
    [
        ("leaf", []),
        ("single_step", [1]),
        ("empty_stock", []),
    ],
)
def test_pruned_route_generation_edge_cases(
    case: str,
    expected_depths: list[int],
) -> None:
    if case == "leaf":
        route = Route(target=molecule("C"))
        stock = {inchikey("C")}
    elif case == "single_step":
        route = linear_route(depth=1)
        stock = {inchikey("C"), inchikey("CC")}
    else:
        route = linear_route(depth=2)
        stock = set()

    routes = generate_pruned_routes(route, stock)

    assert [candidate.depth() for candidate in routes] == expected_depths


@pytest.mark.unit
@given(depth=st.integers(min_value=1, max_value=8))
def test_linear_routes_with_all_intermediates_in_stock_generate_one_route_per_antichain(depth: int) -> None:
    route = linear_route(depth=depth)
    stock = {inchikey("C" * size) for size in range(1, depth + 2)}

    routes = generate_pruned_routes(route, stock)

    assert len(routes) == depth
    assert sorted(candidate.depth() for candidate in routes) == list(range(1, depth + 1))


@pytest.mark.unit
@given(
    depth=st.integers(min_value=1, max_value=8),
    stocked_intermediates=st.integers(min_value=0, max_value=5),
)
def test_pruned_linear_routes_preserve_core_invariants(depth: int, stocked_intermediates: int) -> None:
    route = linear_route(depth=depth)
    stock = {inchikey("C")}
    for size in range(2, min(depth + 1, stocked_intermediates + 2)):
        stock.add(inchikey("C" * size))

    routes = generate_pruned_routes(route, stock)

    assert len({candidate.signature() for candidate in routes}) == len(routes)
    for candidate in routes:
        assert candidate.target.smiles == route.target.smiles
        assert candidate.target.inchikey == route.target.inchikey
        assert candidate.depth() <= route.depth()
        assert route_leaves_are_in_stock(candidate, stock)


def linear_route(depth: int) -> Route:
    if depth < 1:
        raise ValueError("depth must be at least 1")

    current = molecule("C")
    for size in range(2, depth + 2):
        current = molecule("C" * size, product_of=Reaction(reactants=[current]))
    return Route(target=current)


def convergent_route() -> Route:
    left_low = molecule("CC", product_of=Reaction(reactants=[molecule("C")]))
    left_high = molecule("CCC", product_of=Reaction(reactants=[left_low]))
    right = molecule("CO", product_of=Reaction(reactants=[molecule("O")]))
    return Route(target=molecule("CCO", product_of=Reaction(reactants=[left_high, right])))


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(smiles=SmilesStr(canonical), inchikey=inchikey(canonical), product_of=product_of)


def inchikey(smiles: str) -> InChIKeyStr:
    return InChIKeyStr(get_inchi_key(canonicalize_smiles(smiles)))


def route_leaf_smiles(route: Route) -> set[str]:
    return {str(leaf.value.smiles) for leaf in route.iter_leaves()}


def route_leaves_are_in_stock(route: Route, stock: set[InChIKeyStr]) -> bool:
    stock_keys = {str(key) for key in stock}
    return all(str(leaf.value.inchikey) in stock_keys for leaf in route.iter_leaves())
