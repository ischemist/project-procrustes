# tests/domain/test_tree.py

import random

from retrocast.domain.DEPRECATE_schemas import BenchmarkTree, MoleculeNode, ReactionNode, TargetInput
from retrocast.domain.tree import (
    calculate_route_length,
    deduplicate_routes,
    sample_k_by_length,
    sample_random_k,
    sample_top_k,
)
from retrocast.utils.hashing import generate_molecule_hash


def _build_simple_tree(target_smiles: str, reactant_smiles_list: list[str]) -> BenchmarkTree:
    """A helper function to quickly build a 1-step BenchmarkTree for testing."""
    target_info = TargetInput(id=target_smiles, smiles=target_smiles)
    reactants = []
    for i, smiles in enumerate(reactant_smiles_list):
        reactants.append(
            MoleculeNode(
                id=f"root-{i}",
                molecule_hash=generate_molecule_hash(smiles),
                smiles=smiles,
                is_starting_material=True,
                reactions=[],
            )
        )

    reaction = ReactionNode(
        id="rxn-root",
        reaction_smiles=f"{'.'.join(sorted(r.smiles for r in reactants))}>>{target_smiles}",
        reactants=reactants,
    )

    root_node = MoleculeNode(
        id="root",
        molecule_hash=generate_molecule_hash(target_smiles),
        smiles=target_smiles,
        is_starting_material=False,
        reactions=[reaction],
    )

    return BenchmarkTree(target=target_info, retrosynthetic_tree=root_node)


def _build_route_of_length(target_id: str, length: int) -> BenchmarkTree:
    """Builds a linear route of a specific length for testing filtering."""
    if length == 0:
        target_info = TargetInput(id=target_id, smiles=target_id)
        node = MoleculeNode(
            id="root",
            molecule_hash=generate_molecule_hash(target_id),
            smiles=target_id,
            is_starting_material=True,
        )
        return BenchmarkTree(target=target_info, retrosynthetic_tree=node)

    # Start from the bottom up
    reactant1 = MoleculeNode(
        id=f"sm1-{target_id}",
        molecule_hash=generate_molecule_hash(f"sm1-{target_id}"),
        smiles=f"sm1-{target_id}",
        is_starting_material=True,
    )
    reactant2 = MoleculeNode(
        id=f"sm2-{target_id}",
        molecule_hash=generate_molecule_hash(f"sm2-{target_id}"),
        smiles=f"sm2-{target_id}",
        is_starting_material=True,
    )

    current_product_smiles = f"inter-0-{target_id}"
    reaction = ReactionNode(
        id=f"rxn-0-{target_id}",
        reaction_smiles=f"{reactant1.smiles}.{reactant2.smiles}>>{current_product_smiles}",
        reactants=[reactant1, reactant2],
    )
    product_node = MoleculeNode(
        id=f"node-0-{target_id}",
        molecule_hash=generate_molecule_hash(current_product_smiles),
        smiles=current_product_smiles,
        is_starting_material=False,
        reactions=[reaction],
    )

    for i in range(1, length):
        reactant = MoleculeNode(
            id=f"sm-{i + 2}-{target_id}",
            molecule_hash=generate_molecule_hash(f"sm-{i + 2}-{target_id}"),
            smiles=f"sm-{i + 2}-{target_id}",
            is_starting_material=True,
        )
        prev_product_smiles = product_node.smiles
        current_product_smiles = f"inter-{i}-{target_id}" if i < length - 1 else target_id

        reaction = ReactionNode(
            id=f"rxn-{i}-{target_id}",
            reaction_smiles=f"{prev_product_smiles}.{reactant.smiles}>>{current_product_smiles}",
            reactants=[product_node, reactant],
        )
        product_node = MoleculeNode(
            id=f"node-{i}-{target_id}",
            molecule_hash=generate_molecule_hash(current_product_smiles),
            smiles=current_product_smiles,
            is_starting_material=False,
            reactions=[reaction],
        )

    target_info = TargetInput(id=target_id, smiles=target_id)
    return BenchmarkTree(target=target_info, retrosynthetic_tree=product_node)


# --- Deduplication Tests ---


def test_deduplicate_keeps_unique_routes() -> None:
    route1 = _build_simple_tree("T", ["A", "B"])
    route2 = _build_simple_tree("T", ["C", "D"])
    assert len(deduplicate_routes([route1, route2])) == 2


def test_deduplicate_removes_identical_routes() -> None:
    route1 = _build_simple_tree("T", ["A", "B"])
    route2 = _build_simple_tree("T", ["A", "B"])
    assert len(deduplicate_routes([route1, route2])) == 1


def test_deduplicate_removes_reactant_order_duplicates() -> None:
    route1 = _build_simple_tree("T", ["A", "B"])
    route2 = _build_simple_tree("T", ["B", "A"])
    assert len(deduplicate_routes([route1, route2])) == 1


def test_deduplicate_distinguishes_different_assembly_order() -> None:
    """Tests (A+B>>I1)+C>>T is different from A+(B+C>>I2)>>T"""
    target_info = TargetInput(id="T", smiles="T")
    # Route 1
    i1_tree = _build_simple_tree("I1", ["A", "B"]).retrosynthetic_tree
    c_node = MoleculeNode(id="c", molecule_hash=generate_molecule_hash("C"), smiles="C", is_starting_material=True)
    r1_final = ReactionNode(id="r1-final", reaction_smiles="C.I1>>T", reactants=[i1_tree, c_node])
    r1_root = MoleculeNode(
        id="r1-root",
        molecule_hash=generate_molecule_hash("T"),
        smiles="T",
        is_starting_material=False,
        reactions=[r1_final],
    )
    route1 = BenchmarkTree(target=target_info, retrosynthetic_tree=r1_root)
    # Route 2
    i2_tree = _build_simple_tree("I2", ["B", "C"]).retrosynthetic_tree
    a_node = MoleculeNode(id="a", molecule_hash=generate_molecule_hash("A"), smiles="A", is_starting_material=True)
    r2_final = ReactionNode(id="r2-final", reaction_smiles="A.I2>>T", reactants=[i2_tree, a_node])
    r2_root = MoleculeNode(
        id="r2-root",
        molecule_hash=generate_molecule_hash("T"),
        smiles="T",
        is_starting_material=False,
        reactions=[r2_final],
    )
    route2 = BenchmarkTree(target=target_info, retrosynthetic_tree=r2_root)

    assert len(deduplicate_routes([route1, route2])) == 2


# --- Route Length Calculation Tests ---


def test_calculate_route_length_zero_step() -> None:
    tree = _build_route_of_length("A", 0)
    assert calculate_route_length(tree.retrosynthetic_tree) == 0


def test_calculate_route_length_one_step() -> None:
    tree = _build_route_of_length("T", 1)
    assert calculate_route_length(tree.retrosynthetic_tree) == 1


def test_calculate_route_length_multi_step() -> None:
    tree = _build_route_of_length("G", 3)
    assert calculate_route_length(tree.retrosynthetic_tree) == 3


# --- Test sample_top_k ---


def test_sample_top_k_selects_first_k() -> None:
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(10)]
    k = 5
    result = sample_top_k(routes, k)
    assert len(result) == 5
    assert result == routes[:5]


def test_sample_top_k_k_larger_than_list() -> None:
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(3)]
    k = 5
    result = sample_top_k(routes, k)
    assert len(result) == 3
    assert result == routes


def test_sample_top_k_zero_k() -> None:
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(5)]
    assert sample_top_k(routes, 0) == []


def test_sample_top_k_empty_list() -> None:
    assert sample_top_k([], 5) == []


# --- Test sample_random_k ---


def test_sample_random_k_selects_k_items() -> None:
    random.seed(42)  # for reproducibility
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(20)]
    k = 10
    result = sample_random_k(routes, k)
    assert len(result) == k
    result_smiles = {r.target.smiles for r in result}
    original_smiles = {r.target.smiles for r in routes}
    assert result_smiles.issubset(original_smiles)


def test_sample_random_k_k_larger_than_list() -> None:
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(5)]
    k = 10
    result = sample_random_k(routes, k)
    assert len(result) == 5
    assert {r.target.id for r in result} == {r.target.id for r in routes}


def test_sample_random_k_zero_k() -> None:
    routes = [_build_simple_tree(f"T{i}", ["A", "B"]) for i in range(5)]
    assert sample_random_k(routes, 0) == []


def test_sample_random_k_empty_list() -> None:
    assert sample_random_k([], 5) == []


# --- Test sample_k_by_length ---


def test_sample_k_by_length_basic_round_robin() -> None:
    """Tests the round-robin selection works as expected."""
    routes = [
        _build_route_of_length("L1-R1", 1),
        _build_route_of_length("L1-R2", 1),
        _build_route_of_length("L2-R1", 2),
        _build_route_of_length("L2-R2", 2),
        _build_route_of_length("L3-R1", 3),
    ]
    k = 4
    result = sample_k_by_length(routes, k)
    assert len(result) == 4
    lengths = [calculate_route_length(r.retrosynthetic_tree) for r in result]
    assert lengths.count(1) == 2
    assert lengths.count(2) == 1
    assert lengths.count(3) == 1


def test_sample_k_by_length_users_scenario() -> None:
    """Tests the 10-route budget with 3 length groups, expecting a 4/3/3 split."""
    routes = (
        [_build_route_of_length(f"L5-R{i}", 5) for i in range(10)]
        + [_build_route_of_length(f"L6-R{i}", 6) for i in range(10)]
        + [_build_route_of_length(f"L7-R{i}", 7) for i in range(10)]
    )
    k = 10
    result = sample_k_by_length(routes, k)
    assert len(result) == 10
    lengths = [calculate_route_length(r.retrosynthetic_tree) for r in result]
    assert lengths.count(5) == 4
    assert lengths.count(6) == 3
    assert lengths.count(7) == 3


def test_sample_k_by_length_uneven_distribution() -> None:
    """Tests when some length groups are exhausted before others."""
    routes = [
        _build_route_of_length("L1-R1", 1),
        _build_route_of_length("L2-R1", 2),
        _build_route_of_length("L2-R2", 2),
        _build_route_of_length("L3-R1", 3),
        _build_route_of_length("L3-R2", 3),
        _build_route_of_length("L3-R3", 3),
    ]
    k = 5
    result = sample_k_by_length(routes, k)
    assert len(result) == 5
    lengths = [calculate_route_length(r.retrosynthetic_tree) for r in result]
    assert lengths.count(1) == 1
    assert lengths.count(2) == 2
    assert lengths.count(3) == 2


def test_sample_k_by_length_k_larger_than_list() -> None:
    routes = [_build_route_of_length("L1-R1", 1), _build_route_of_length("L2-R1", 2)]
    k = 5
    result = sample_k_by_length(routes, k)
    assert len(result) == 2
    assert {r.target.id for r in result} == {"L1-R1", "L2-R1"}


def test_sample_k_by_length_zero_k() -> None:
    routes = [_build_route_of_length(f"T{i}", 1) for i in range(5)]
    assert sample_k_by_length(routes, 0) == []


def test_sample_k_by_length_empty_list() -> None:
    assert sample_k_by_length([], 5) == []
