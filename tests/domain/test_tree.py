# tests/domain/test_tree.py

import random

from retrocast.domain.chem import get_inchi_key
from retrocast.domain.tree import (
    deduplicate_routes,
    excise_reactions_from_route,
    sample_k_by_depth,
    sample_random_k,
    sample_top_k,
)
from retrocast.schemas import Molecule, ReactionSignature, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr


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


# --- Excise Reactions Tests ---


def test_excise_reactions_empty_exclusion_set() -> None:
    """When no reactions to exclude, should return original route unchanged."""
    route = _build_simple_route("CO", ["CCO", "CCCO"])
    result = excise_reactions_from_route(route, set())

    assert len(result) == 1
    assert result[0].get_signature() == route.get_signature()


def test_excise_reactions_leaf_route() -> None:
    """Leaf route (no reactions) with any exclusion set should return empty."""
    leaf = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
    )
    route = Route(target=leaf, rank=1)

    # Exclude some random reaction
    exclude: set[ReactionSignature] = {(frozenset(["KEY-A"]), "KEY-B")}
    result = excise_reactions_from_route(route, exclude)

    # No reactions in route, so nothing to excise, but route has no depth
    assert len(result) == 0


def test_excise_reactions_single_step_route_excise_only_reaction() -> None:
    """Single step route: excising the only reaction leaves no valid sub-routes."""
    route = _build_simple_route("CO", ["CCO", "CCCO"])

    # Get the signature of the only reaction
    sigs = route.get_reaction_signatures()
    assert len(sigs) == 1

    result = excise_reactions_from_route(route, sigs)

    # The main route becomes a leaf (no reactions), so no valid sub-routes
    assert len(result) == 0


def test_excise_reactions_linear_route_excise_middle() -> None:
    """
    Linear route: A -> B -> C (target)
    Excise A -> B, should leave: B -> C (with B as leaf)
    """
    leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
    intermediate_b = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B"),
        synthesis_step=ReactionStep(reactants=[leaf_a]),
    )
    target_c = Molecule(
        smiles=SmilesStr("COC"),
        inchikey=InchiKeyStr("KEY-C"),
        synthesis_step=ReactionStep(reactants=[intermediate_b]),
    )
    route = Route(target=target_c, rank=1)

    # Excise the first reaction: A -> B
    exclude: set[ReactionSignature] = {(frozenset(["KEY-A"]), "KEY-B")}
    result = excise_reactions_from_route(route, exclude)

    # Should have 1 sub-route: C with B as a leaf
    assert len(result) == 1
    main_route = result[0]
    assert main_route.target.inchikey == "KEY-C"
    assert main_route.depth == 1  # Only one reaction left

    # B should now be a leaf
    reactants = main_route.target.synthesis_step.reactants
    assert len(reactants) == 1
    assert reactants[0].inchikey == "KEY-B"
    assert reactants[0].is_leaf


def test_excise_reactions_linear_route_excise_last() -> None:
    """
    Linear route: A -> B -> C (target)
    Excise B -> C, should leave: A -> B (as separate sub-route)
    """
    leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
    intermediate_b = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B"),
        synthesis_step=ReactionStep(reactants=[leaf_a]),
    )
    target_c = Molecule(
        smiles=SmilesStr("COC"),
        inchikey=InchiKeyStr("KEY-C"),
        synthesis_step=ReactionStep(reactants=[intermediate_b]),
    )
    route = Route(target=target_c, rank=1)

    # Excise the last reaction: B -> C
    exclude: set[ReactionSignature] = {(frozenset(["KEY-B"]), "KEY-C")}
    result = excise_reactions_from_route(route, exclude)

    # Main route becomes leaf (no valid reactions), but B becomes a sub-route
    assert len(result) == 1
    sub_route = result[0]
    assert sub_route.target.inchikey == "KEY-B"
    assert sub_route.depth == 1

    # Verify A -> B reaction is preserved
    reactants = sub_route.target.synthesis_step.reactants
    assert len(reactants) == 1
    assert reactants[0].inchikey == "KEY-A"


def test_excise_reactions_branched_route() -> None:
    """
    Branched route:
           C (target)
          /  \
         B1   B2
         |    |
         A1   A2
    
    Excise A1 -> B1, should leave C with B1 as leaf and B2 with synthesis
    """
    leaf_a1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A1"))
    leaf_a2 = Molecule(smiles=SmilesStr("CC"), inchikey=InchiKeyStr("KEY-A2"))

    intermediate_b1 = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B1"),
        synthesis_step=ReactionStep(reactants=[leaf_a1]),
    )
    intermediate_b2 = Molecule(
        smiles=SmilesStr("CCO"),
        inchikey=InchiKeyStr("KEY-B2"),
        synthesis_step=ReactionStep(reactants=[leaf_a2]),
    )

    target_c = Molecule(
        smiles=SmilesStr("CCOC"),
        inchikey=InchiKeyStr("KEY-C"),
        synthesis_step=ReactionStep(reactants=[intermediate_b1, intermediate_b2]),
    )
    route = Route(target=target_c, rank=1)

    # Excise A1 -> B1
    exclude: set[ReactionSignature] = {(frozenset(["KEY-A1"]), "KEY-B1")}
    result = excise_reactions_from_route(route, exclude)

    # Should have 1 sub-route: C with B1 as leaf but B2 with synthesis
    assert len(result) == 1
    main_route = result[0]
    assert main_route.target.inchikey == "KEY-C"
    assert main_route.depth == 2  # Still has A2 -> B2 -> C

    # B1 should be a leaf, B2 should have synthesis
    reactants = main_route.target.synthesis_step.reactants
    b1_node = next(r for r in reactants if r.inchikey == "KEY-B1")
    b2_node = next(r for r in reactants if r.inchikey == "KEY-B2")

    assert b1_node.is_leaf
    assert not b2_node.is_leaf
    assert b2_node.synthesis_step.reactants[0].inchikey == "KEY-A2"


def test_excise_reactions_creates_multiple_subroutes() -> None:
    """
    Linear route: A -> B -> C -> D (target)
    Excise B -> C, should create two sub-routes: A -> B and C -> D
    """
    leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
    node_b = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B"),
        synthesis_step=ReactionStep(reactants=[leaf_a]),
    )
    node_c = Molecule(
        smiles=SmilesStr("COC"),
        inchikey=InchiKeyStr("KEY-C"),
        synthesis_step=ReactionStep(reactants=[node_b]),
    )
    target_d = Molecule(
        smiles=SmilesStr("COCC"),
        inchikey=InchiKeyStr("KEY-D"),
        synthesis_step=ReactionStep(reactants=[node_c]),
    )
    route = Route(target=target_d, rank=1)

    # Excise B -> C
    exclude: set[ReactionSignature] = {(frozenset(["KEY-B"]), "KEY-C")}
    result = excise_reactions_from_route(route, exclude)

    # Should have 2 sub-routes
    assert len(result) == 2

    # Main route: D with C as leaf
    main_route = next(r for r in result if r.target.inchikey == "KEY-D")
    assert main_route.depth == 1
    assert main_route.target.synthesis_step.reactants[0].inchikey == "KEY-C"
    assert main_route.target.synthesis_step.reactants[0].is_leaf

    # Sub-route: B with A as leaf
    sub_route = next(r for r in result if r.target.inchikey == "KEY-B")
    assert sub_route.depth == 1
    assert sub_route.target.synthesis_step.reactants[0].inchikey == "KEY-A"


def test_excise_reactions_preserves_metadata() -> None:
    """Metadata should be preserved on molecules and reactions."""
    leaf = Molecule(
        smiles=SmilesStr("C"),
        inchikey=InchiKeyStr("KEY-A"),
        metadata={"leaf_data": "preserved"},
    )
    step = ReactionStep(
        reactants=[leaf],
        mapped_smiles="C>>CO",
        template="[C:1]>>[C:1]O",
        reagents=["O"],
        metadata={"rxn_score": 0.95},
    )
    target = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B"),
        synthesis_step=step,
        metadata={"target_data": "also_preserved"},
    )
    route = Route(target=target, rank=1, metadata={"route_info": "test"})

    result = excise_reactions_from_route(route, set())

    assert len(result) == 1
    new_route = result[0]

    # Route metadata preserved
    assert new_route.metadata == {"route_info": "test"}
    assert new_route.rank == 1

    # Target metadata preserved
    assert new_route.target.metadata == {"target_data": "also_preserved"}

    # Reaction metadata preserved
    assert new_route.target.synthesis_step.mapped_smiles == "C>>CO"
    assert new_route.target.synthesis_step.template == "[C:1]>>[C:1]O"
    assert new_route.target.synthesis_step.reagents == ["O"]
    assert new_route.target.synthesis_step.metadata == {"rxn_score": 0.95}

    # Leaf metadata preserved
    leaf_mol = new_route.target.synthesis_step.reactants[0]
    assert leaf_mol.metadata == {"leaf_data": "preserved"}


def test_excise_multiple_reactions() -> None:
    """Excise multiple reactions from the same route."""
    # Build: A -> B -> C -> D
    leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
    node_b = Molecule(
        smiles=SmilesStr("CO"),
        inchikey=InchiKeyStr("KEY-B"),
        synthesis_step=ReactionStep(reactants=[leaf_a]),
    )
    node_c = Molecule(
        smiles=SmilesStr("COC"),
        inchikey=InchiKeyStr("KEY-C"),
        synthesis_step=ReactionStep(reactants=[node_b]),
    )
    target_d = Molecule(
        smiles=SmilesStr("COCC"),
        inchikey=InchiKeyStr("KEY-D"),
        synthesis_step=ReactionStep(reactants=[node_c]),
    )
    route = Route(target=target_d, rank=1)

    # Excise both A -> B and C -> D
    exclude: set[ReactionSignature] = {
        (frozenset(["KEY-A"]), "KEY-B"),
        (frozenset(["KEY-C"]), "KEY-D"),
    }
    result = excise_reactions_from_route(route, exclude)

    # Main route (D) has no reactions (C is a leaf), so it's excluded
    # C has B -> C reaction, but B is now a leaf (A -> B was excised)
    # So we should have 1 sub-route: B -> C with B as leaf
    assert len(result) == 1
    sub_route = result[0]
    assert sub_route.target.inchikey == "KEY-C"
    assert sub_route.depth == 1
