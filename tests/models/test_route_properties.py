"""Property-based tests for Route using hypothesis.

These tests verify fundamental invariants of the Route class that must
hold for all possible inputs, not just manually-crafted examples.
"""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr

# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def synthetic_inchikey(draw):
    """Generate a synthetic InchiKey-like string."""
    # Format: XXXXXXXXXXXXXX-XXXXXXXXXX-X
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    part1 = "".join(draw(st.sampled_from(chars)) for _ in range(14))
    part2 = "".join(draw(st.sampled_from(chars)) for _ in range(10))
    part3 = draw(st.sampled_from("NMO"))
    return f"{part1}-{part2}-{part3}"


@st.composite
def leaf_molecule(draw):
    """Generate a leaf molecule with synthetic data."""
    smiles = draw(st.sampled_from(["C", "CC", "CCC", "O", "N", "CO", "CCO"]))
    inchikey = draw(synthetic_inchikey())
    return Molecule(
        smiles=SmilesStr(smiles),
        inchikey=InchiKeyStr(inchikey),
    )


@st.composite
def simple_route_with_reactants(draw, min_reactants=2, max_reactants=5):
    """Generate a route with a single reaction step and multiple reactants."""
    num_reactants = draw(st.integers(min_value=min_reactants, max_value=max_reactants))
    reactants = [draw(leaf_molecule()) for _ in range(num_reactants)]

    # Create product
    product_smiles = "C" * (num_reactants + 1)
    product_inchikey = draw(synthetic_inchikey())

    step = ReactionStep(reactants=reactants)
    target = Molecule(
        smiles=SmilesStr(product_smiles),
        inchikey=InchiKeyStr(product_inchikey),
        synthesis_step=step,
    )

    return Route(target=target, rank=1), reactants


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestRouteSignatureProperties:
    """Property-based tests for Route.get_signature()."""

    @given(simple_route_with_reactants())
    @settings(max_examples=100)
    @pytest.mark.unit
    def test_signature_order_invariant(self, route_and_reactants):
        """Test that get_signature is invariant to reactant order (commutative)."""
        route, reactants = route_and_reactants
        original_signature = route.get_signature()

        # Shuffle reactants and rebuild route
        shuffled_reactants = reactants.copy()
        random.shuffle(shuffled_reactants)

        shuffled_step = ReactionStep(reactants=shuffled_reactants)
        shuffled_target = Molecule(
            smiles=route.target.smiles,
            inchikey=route.target.inchikey,
            synthesis_step=shuffled_step,
        )
        shuffled_route = Route(target=shuffled_target, rank=route.rank)
        shuffled_signature = shuffled_route.get_signature()

        assert original_signature == shuffled_signature, "Signature should be identical regardless of reactant order"

    @given(simple_route_with_reactants())
    @settings(max_examples=50)
    @pytest.mark.unit
    def test_signature_deterministic(self, route_and_reactants):
        """Test that get_signature always returns the same value for same route."""
        route, _ = route_and_reactants

        sig1 = route.get_signature()
        sig2 = route.get_signature()
        sig3 = route.get_signature()

        assert sig1 == sig2 == sig3, "Signature must be deterministic"

    @given(simple_route_with_reactants())
    @settings(max_examples=50)
    @pytest.mark.unit
    def test_signature_is_valid_hash(self, route_and_reactants):
        """Test that get_signature returns a valid SHA256 hex string."""
        route, _ = route_and_reactants
        signature = route.get_signature()

        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in signature)

    @pytest.mark.unit
    def test_signature_permutation_invariant_swap_children(self):
        """Test that swapping left/right children does not change signature.

        This explicitly tests that routes like:
            target <- [A, B]  and  target <- [B, A]
        produce the same signature.
        """
        # Create two leaf molecules
        leaf_a = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("AAAAAAAAAAAAA-AAAAAAAAAA-N"),
        )
        leaf_b = Molecule(
            smiles=SmilesStr("CC"),
            inchikey=InchiKeyStr("BBBBBBBBBBBBB-BBBBBBBBBB-N"),
        )

        # Route 1: target <- [A, B]
        step1 = ReactionStep(reactants=[leaf_a, leaf_b])
        target1 = Molecule(
            smiles=SmilesStr("CCC"),
            inchikey=InchiKeyStr("CCCCCCCCCCCCC-CCCCCCCCCC-N"),
            synthesis_step=step1,
        )
        route1 = Route(target=target1, rank=1)

        # Route 2: target <- [B, A] (swapped)
        step2 = ReactionStep(reactants=[leaf_b, leaf_a])
        target2 = Molecule(
            smiles=SmilesStr("CCC"),
            inchikey=InchiKeyStr("CCCCCCCCCCCCC-CCCCCCCCCC-N"),
            synthesis_step=step2,
        )
        route2 = Route(target=target2, rank=1)

        assert route1.get_signature() == route2.get_signature(), (
            "Swapping left/right children should not change signature"
        )

    @pytest.mark.unit
    def test_signature_permutation_invariant_deep_tree(self):
        """Test permutation invariance in a deeper tree structure.

        Tests that swapping children at different levels doesn't affect signature.
        """
        # Create leaves
        leaf_1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("LEAF1-INCHIKEY-N"))
        leaf_2 = Molecule(smiles=SmilesStr("O"), inchikey=InchiKeyStr("LEAF2-INCHIKEY-N"))
        leaf_3 = Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("LEAF3-INCHIKEY-N"))

        # Build intermediate from leaf_1 and leaf_2
        intermediate_a = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("INTER-A-INCHKEY-N"),
            synthesis_step=ReactionStep(reactants=[leaf_1, leaf_2]),
        )

        # Build intermediate with swapped order
        intermediate_a_swapped = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("INTER-A-INCHKEY-N"),
            synthesis_step=ReactionStep(reactants=[leaf_2, leaf_1]),  # Swapped
        )

        # Build final targets
        target1 = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("TARGET-INCHKEY-N"),
            synthesis_step=ReactionStep(reactants=[intermediate_a, leaf_3]),
        )
        route1 = Route(target=target1, rank=1)

        # Target with swapped intermediate reactants
        target2 = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("TARGET-INCHKEY-N"),
            synthesis_step=ReactionStep(reactants=[intermediate_a_swapped, leaf_3]),
        )
        route2 = Route(target=target2, rank=1)

        # Target with swapped top-level reactants
        target3 = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("TARGET-INCHKEY-N"),
            synthesis_step=ReactionStep(reactants=[leaf_3, intermediate_a]),  # Swapped
        )
        route3 = Route(target=target3, rank=1)

        sig1 = route1.get_signature()
        sig2 = route2.get_signature()
        sig3 = route3.get_signature()

        assert sig1 == sig2, "Swapping reactants at intermediate level should not change signature"
        assert sig1 == sig3, "Swapping reactants at top level should not change signature"

    @pytest.mark.unit
    def test_signature_permutation_invariant_three_reactants(self):
        """Test permutation invariance with three reactants (all permutations)."""
        import itertools

        leaves = [
            Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("AAA-INCHIKEY-N")),
            Molecule(smiles=SmilesStr("O"), inchikey=InchiKeyStr("BBB-INCHIKEY-N")),
            Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("CCC-INCHIKEY-N")),
        ]

        signatures = []
        for perm in itertools.permutations(leaves):
            step = ReactionStep(reactants=list(perm))
            target = Molecule(
                smiles=SmilesStr("CON"),
                inchikey=InchiKeyStr("TARGET-INCHIKEY-N"),
                synthesis_step=step,
            )
            route = Route(target=target, rank=1)
            signatures.append(route.get_signature())

        # All 6 permutations should produce the same signature
        assert len(set(signatures)) == 1, (
            f"All permutations should produce same signature, got {len(set(signatures))} unique signatures"
        )


class TestRouteContentHashProperties:
    """Property-based tests for Route.get_content_hash()."""

    @given(simple_route_with_reactants())
    @settings(max_examples=50)
    @pytest.mark.unit
    def test_content_hash_deterministic(self, route_and_reactants):
        """Test that get_content_hash is deterministic."""
        route, _ = route_and_reactants

        hash1 = route.get_content_hash()
        hash2 = route.get_content_hash()

        assert hash1 == hash2, "Content hash must be deterministic"

    @given(simple_route_with_reactants(), st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    @pytest.mark.unit
    def test_content_hash_changes_with_rank(self, route_and_reactants, new_rank):
        """Test that content hash differs when rank changes."""
        route, _ = route_and_reactants

        # Create route with different rank
        different_rank_route = Route(
            target=route.target,
            rank=new_rank if new_rank != route.rank else new_rank + 1,
        )

        # Content hash should differ
        assert route.get_content_hash() != different_rank_route.get_content_hash()

        # But tree signature should be the same
        assert route.get_signature() == different_rank_route.get_signature()


class TestRouteTopologyProperties:
    """Property-based tests for route topology calculations."""

    @pytest.mark.unit
    def test_linear_route_depth_equals_factory_depth(self, synthetic_route_factory):
        """Test that linear routes have correct depth."""
        for depth in range(1, 6):
            route = synthetic_route_factory("linear", depth=depth)
            assert route.length == depth, f"Linear route with depth={depth} should have length={depth}"

    @pytest.mark.unit
    def test_linear_route_has_single_leaf(self, synthetic_route_factory):
        """Test that linear routes have exactly one leaf."""
        for depth in range(1, 6):
            route = synthetic_route_factory("linear", depth=depth)
            assert len(route.leaves) == 1, "Linear route should have exactly 1 leaf"

    @pytest.mark.unit
    def test_linear_route_not_convergent(self, synthetic_route_factory):
        """Test that linear routes are not convergent."""
        for depth in range(1, 6):
            route = synthetic_route_factory("linear", depth=depth)
            assert route.has_convergent_reaction is False, "Linear routes should not be convergent"

    @pytest.mark.unit
    def test_convergent_route_is_convergent(self, synthetic_route_factory):
        """Test that convergent routes are detected as convergent."""
        for depth in range(2, 6):
            route = synthetic_route_factory("convergent", depth=depth)
            assert route.has_convergent_reaction is True, f"Convergent route with depth={depth} should be convergent"

    @pytest.mark.unit
    def test_convergent_route_has_multiple_leaves(self, synthetic_route_factory):
        """Test that convergent routes have multiple leaves."""
        for depth in range(2, 6):
            route = synthetic_route_factory("convergent", depth=depth)
            assert len(route.leaves) >= 2, "Convergent route should have at least 2 leaves"

    @pytest.mark.unit
    def test_binary_tree_route_is_convergent(self, synthetic_route_factory):
        """Test that binary tree routes with depth >= 2 are convergent.

        Note: depth=1 combines only leaves (not intermediates), so it's not convergent
        by our definition. Depth >= 2 combines intermediates.
        """
        for depth in range(2, 4):
            route = synthetic_route_factory("binary_tree", depth=depth)
            assert route.has_convergent_reaction is True, f"Binary tree with depth={depth} should be convergent"

    @pytest.mark.unit
    def test_binary_tree_leaf_count(self, synthetic_route_factory):
        """Test that binary tree routes have 2^depth leaves."""
        for depth in range(1, 4):
            route = synthetic_route_factory("binary_tree", depth=depth)
            expected_leaves = 2**depth
            assert len(route.leaves) == expected_leaves, (
                f"Binary tree with depth={depth} should have {expected_leaves} leaves"
            )


class TestSyntheticRouteFactory:
    """Tests for the synthetic route factory fixture itself."""

    @pytest.mark.unit
    def test_factory_produces_valid_routes(self, synthetic_route_factory):
        """Test that factory produces valid Route objects."""
        for structure in ["linear", "convergent", "binary_tree"]:
            depth = 2 if structure == "convergent" else 1
            route = synthetic_route_factory(structure, depth=depth)

            assert isinstance(route, Route)
            assert route.rank == 1
            assert route.target is not None
            assert isinstance(route.get_signature(), str)

    @pytest.mark.unit
    def test_factory_linear_invalid_depth_raises(self, synthetic_route_factory):
        """Test that invalid depth raises ValueError."""
        with pytest.raises(ValueError):
            synthetic_route_factory("linear", depth=0)

    @pytest.mark.unit
    def test_factory_convergent_invalid_depth_raises(self, synthetic_route_factory):
        """Test that convergent with depth < 2 raises ValueError."""
        with pytest.raises(ValueError):
            synthetic_route_factory("convergent", depth=1)

    @pytest.mark.unit
    def test_factory_unknown_structure_raises(self, synthetic_route_factory):
        """Test that unknown structure raises ValueError."""
        with pytest.raises(ValueError, match="Unknown structure"):
            synthetic_route_factory("unknown", depth=2)

    @pytest.mark.unit
    def test_factory_routes_have_deterministic_signatures(self, synthetic_route_factory):
        """Test that routes from factory have deterministic signatures."""
        route1 = synthetic_route_factory("linear", depth=3)
        route2 = synthetic_route_factory("linear", depth=3)

        # Same parameters should produce routes with same signature
        assert route1.get_signature() == route2.get_signature()
