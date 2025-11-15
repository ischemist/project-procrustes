# tests/domain/test_tree.py

import random

from retrocast.domain.chem import get_inchi_key
from retrocast.domain.tree import (
    deduplicate_routes,
    sample_k_by_depth,
    sample_random_k,
    sample_top_k,
)
from retrocast.schemas import Molecule, ReactionStep, Route


def _build_simple_route(target_smiles: str, reactant_smiles_list: list[str]) -> Route:
    """A helper function to quickly build a 1-step Route for testing."""
    reactants = []
    for smiles in reactant_smiles_list:
        reactants.append(
            Molecule(
                smiles=smiles,
                inchikey=get_inchi_key(smiles),
                synthesis_step=None,
            )
        )

    reaction = ReactionStep(reactants=reactants)
    root_molecule = Molecule(
        smiles=target_smiles,
        inchikey=get_inchi_key(target_smiles),
        synthesis_step=reaction,
    )

    return Route(target=root_molecule, rank=1)


def _build_route_of_depth(target_id: str, depth: int) -> Route:
    """Builds a linear route of a specific depth for testing filtering.

    Uses a simple alkane chain strategy where each step adds carbons.
    For a route of depth N, we build backwards from the target.
    """
    if depth == 0:
        # Leaf molecule (no synthesis) - use methanol as a simple valid SMILES
        node = Molecule(
            smiles="CO",
            inchikey=get_inchi_key("CO"),
            synthesis_step=None,
        )
        return Route(target=node, rank=1)

    # We'll use a counter to generate unique valid SMILES for each route
    # Use the target_id hash to make different routes have different molecules
    seed = hash(target_id) % 100

    # Start from the bottom up - use simple alcohols as starting materials
    reactant1 = Molecule(
        smiles="CO",  # methanol
        inchikey=get_inchi_key("CO"),
        synthesis_step=None,
    )
    reactant2 = Molecule(
        smiles="CCO",  # ethanol
        inchikey=get_inchi_key("CCO"),
        synthesis_step=None,
    )

    # First intermediate uses propanol
    current_smiles = "CCCO"
    reaction = ReactionStep(reactants=[reactant1, reactant2])
    product_node = Molecule(
        smiles=current_smiles,
        inchikey=get_inchi_key(current_smiles),
        synthesis_step=reaction,
    )

    # Build up the chain for remaining depth
    for i in range(1, depth):
        # Each step adds one more carbon to the chain
        reactant = Molecule(
            smiles="C" * (i + 2) + "O",  # butanol, pentanol, etc.
            inchikey=get_inchi_key("C" * (i + 2) + "O"),
            synthesis_step=None,
        )

        if i < depth - 1:
            # Intermediate product - longer alcohol
            current_smiles = "C" * (i + 4) + "O"
        else:
            # Final target - use a unique alkane based on seed and depth
            current_smiles = "C" * (seed + depth + i + 5)

        reaction = ReactionStep(reactants=[product_node, reactant])
        product_node = Molecule(
            smiles=current_smiles,
            inchikey=get_inchi_key(current_smiles),
            synthesis_step=reaction,
        )

    return Route(target=product_node, rank=1)


# --- Deduplication Tests ---


def test_deduplicate_keeps_unique_routes() -> None:
    # Use valid SMILES: methanol, ethanol, propanol, butanol
    route1 = _build_simple_route("CO", ["CCO", "CCCO"])
    route2 = _build_simple_route("CO", ["CCCCO", "C"])
    assert len(deduplicate_routes([route1, route2])) == 2


def test_deduplicate_removes_identical_routes() -> None:
    route1 = _build_simple_route("CO", ["CCO", "CCCO"])
    route2 = _build_simple_route("CO", ["CCO", "CCCO"])
    assert len(deduplicate_routes([route1, route2])) == 1


def test_deduplicate_removes_reactant_order_duplicates() -> None:
    route1 = _build_simple_route("CO", ["CCO", "CCCO"])
    route2 = _build_simple_route("CO", ["CCCO", "CCO"])
    assert len(deduplicate_routes([route1, route2])) == 1


def test_deduplicate_distinguishes_different_assembly_order() -> None:
    """Tests (A+B>>I1)+C>>T is different from A+(B+C>>I2)>>T"""
    # Use valid SMILES: methane (C), ethane (CC), propane (CCC), butane (CCCC), pentane (CCCCC)
    # Route 1: (methane+ethane>>propane)+butane>>pentane
    i1_route = _build_simple_route("CCC", ["C", "CC"])
    i1_molecule = i1_route.target
    c_molecule = Molecule(smiles="CCCC", inchikey=get_inchi_key("CCCC"), synthesis_step=None)
    r1_reaction = ReactionStep(reactants=[i1_molecule, c_molecule])
    r1_root = Molecule(smiles="CCCCC", inchikey=get_inchi_key("CCCCC"), synthesis_step=r1_reaction)
    route1 = Route(target=r1_root, rank=1)

    # Route 2: methane+(ethane+butane>>hexane)>>heptane
    i2_route = _build_simple_route("CCCCCC", ["CC", "CCCC"])
    i2_molecule = i2_route.target
    a_molecule = Molecule(smiles="C", inchikey=get_inchi_key("C"), synthesis_step=None)
    r2_reaction = ReactionStep(reactants=[i2_molecule, a_molecule])
    r2_root = Molecule(smiles="CCCCCCC", inchikey=get_inchi_key("CCCCCCC"), synthesis_step=r2_reaction)
    route2 = Route(target=r2_root, rank=1)

    assert len(deduplicate_routes([route1, route2])) == 2


# --- Test sample_top_k ---


def test_sample_top_k_selects_first_k() -> None:
    # Use different valid SMILES for each target
    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]
    routes = [_build_simple_route(smiles[i], ["CO", "CCO"]) for i in range(10)]
    k = 5
    result = sample_top_k(routes, k)
    assert len(result) == 5
    assert result == routes[:5]


def test_sample_top_k_k_larger_than_list() -> None:
    routes = [_build_simple_route(smiles, ["CO", "CCO"]) for smiles in ["C", "CC", "CCC"]]
    k = 5
    result = sample_top_k(routes, k)
    assert len(result) == 3
    assert result == routes


def test_sample_top_k_zero_k() -> None:
    routes = [_build_simple_route(smiles, ["CO", "CCO"]) for smiles in ["C", "CC", "CCC", "CCCC", "CCCCC"]]
    assert sample_top_k(routes, 0) == []


def test_sample_top_k_empty_list() -> None:
    assert sample_top_k([], 5) == []


# --- Test sample_random_k ---


def test_sample_random_k_selects_k_items() -> None:
    random.seed(42)  # for reproducibility
    # Generate 20 different valid SMILES (alkanes)
    alkanes = ["C" + "C" * i for i in range(20)]
    routes = [_build_simple_route(alkanes[i], ["CO", "CCO"]) for i in range(20)]
    k = 10
    result = sample_random_k(routes, k)
    assert len(result) == k
    result_smiles = {r.target.smiles for r in result}
    original_smiles = {r.target.smiles for r in routes}
    assert result_smiles.issubset(original_smiles)


def test_sample_random_k_k_larger_than_list() -> None:
    alkanes = ["C" + "C" * i for i in range(5)]
    routes = [_build_simple_route(alkanes[i], ["CO", "CCO"]) for i in range(5)]
    k = 10
    result = sample_random_k(routes, k)
    assert len(result) == 5
    assert {r.target.smiles for r in result} == {r.target.smiles for r in routes}


def test_sample_random_k_zero_k() -> None:
    alkanes = ["C" + "C" * i for i in range(5)]
    routes = [_build_simple_route(alkanes[i], ["CO", "CCO"]) for i in range(5)]
    assert sample_random_k(routes, 0) == []


def test_sample_random_k_empty_list() -> None:
    assert sample_random_k([], 5) == []


# --- Test sample_k_by_depth ---


def test_sample_k_by_depth_basic_round_robin() -> None:
    """Tests the round-robin selection works as expected."""
    routes = [
        _build_route_of_depth("L1-R1", 1),
        _build_route_of_depth("L1-R2", 1),
        _build_route_of_depth("L2-R1", 2),
        _build_route_of_depth("L2-R2", 2),
        _build_route_of_depth("L3-R1", 3),
    ]
    k = 4
    result = sample_k_by_depth(routes, k)
    assert len(result) == 4
    depths = [r.depth for r in result]
    assert depths.count(1) == 2
    assert depths.count(2) == 1
    assert depths.count(3) == 1


def test_sample_k_by_depth_users_scenario() -> None:
    """Tests the 10-route budget with 3 depth groups, expecting a 4/3/3 split."""
    routes = (
        [_build_route_of_depth(f"L5-R{i}", 5) for i in range(10)]
        + [_build_route_of_depth(f"L6-R{i}", 6) for i in range(10)]
        + [_build_route_of_depth(f"L7-R{i}", 7) for i in range(10)]
    )
    k = 10
    result = sample_k_by_depth(routes, k)
    assert len(result) == 10
    depths = [r.depth for r in result]
    assert depths.count(5) == 4
    assert depths.count(6) == 3
    assert depths.count(7) == 3


def test_sample_k_by_depth_uneven_distribution() -> None:
    """Tests when some depth groups are exhausted before others."""
    routes = [
        _build_route_of_depth("L1-R1", 1),
        _build_route_of_depth("L2-R1", 2),
        _build_route_of_depth("L2-R2", 2),
        _build_route_of_depth("L3-R1", 3),
        _build_route_of_depth("L3-R2", 3),
        _build_route_of_depth("L3-R3", 3),
    ]
    k = 5
    result = sample_k_by_depth(routes, k)
    assert len(result) == 5
    depths = [r.depth for r in result]
    assert depths.count(1) == 1
    assert depths.count(2) == 2
    assert depths.count(3) == 2


def test_sample_k_by_depth_k_larger_than_list() -> None:
    routes = [_build_route_of_depth("L1-R1", 1), _build_route_of_depth("L2-R1", 2)]
    k = 5
    result = sample_k_by_depth(routes, k)
    assert len(result) == 2
    # Just verify we got 2 routes back, don't check specific SMILES since they're generated
    assert len({r.target.smiles for r in result}) == 2


def test_sample_k_by_depth_zero_k() -> None:
    alkanes = ["C" + "C" * i for i in range(5)]
    routes = [_build_route_of_depth(alkanes[i], 1) for i in range(5)]
    assert sample_k_by_depth(routes, 0) == []


def test_sample_k_by_depth_empty_list() -> None:
    assert sample_k_by_depth([], 5) == []
