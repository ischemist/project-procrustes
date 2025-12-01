"""
Unit tests for retrocast.curation.generators module.

Tests route generation functions using synthetic carbon-chain routes
following the "no mocking, synthetic topology" philosophy.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retrocast.curation.generators import generate_pruned_routes, get_stock_intermediates
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr
from tests.helpers import _make_leaf_molecule, _synthetic_inchikey

# =============================================================================
# Helper functions for creating test routes
# =============================================================================


def _make_linear_route_with_intermediates(depth: int) -> Route:
    """
    Create a linear route with specified depth.

    Depth 1: CC <- C
    Depth 2: CCC <- CC <- C
    Depth 3: CCCC <- CCC <- CC <- C

    Returns a route where each intermediate can be identified by its carbon count.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    # Start with leaf: C
    current = _make_leaf_molecule("C")

    # Build up the chain
    for i in range(2, depth + 2):
        smiles = "C" * i
        current = Molecule(
            smiles=SmilesStr(smiles),
            inchikey=InchiKeyStr(_synthetic_inchikey(smiles)),
            synthesis_step=ReactionStep(reactants=[current]),
        )

    return Route(target=current, rank=1)


def _make_convergent_route_with_intermediates() -> Route:
    """
    Create a convergent route for testing:

         CCCCCC (target)
           /    \
        CCCC    CC (intermediates)
         /       |
        CC       C  (leaves)
        |
        C

    This has 2 intermediate nodes: CCCC and the left CC.
    """
    # Left branch: C -> CC -> CCCC
    leaf_left = _make_leaf_molecule("C")
    intermediate_left_1 = Molecule(
        smiles=SmilesStr("CC"),
        inchikey=InchiKeyStr(_synthetic_inchikey("branch_left_CC")),
        synthesis_step=ReactionStep(reactants=[leaf_left]),
    )
    intermediate_left_2 = Molecule(
        smiles=SmilesStr("CCCC"),
        inchikey=InchiKeyStr(_synthetic_inchikey("branch_left_CCCC")),
        synthesis_step=ReactionStep(reactants=[intermediate_left_1]),
    )

    # Right branch: C -> CC
    leaf_right = _make_leaf_molecule("O")  # Use O to make it unique
    intermediate_right = Molecule(
        smiles=SmilesStr("CC"),
        inchikey=InchiKeyStr(_synthetic_inchikey("branch_right_CC")),
        synthesis_step=ReactionStep(reactants=[leaf_right]),
    )

    # Convergent step: combine both branches
    target = Molecule(
        smiles=SmilesStr("CCCCCC"),
        inchikey=InchiKeyStr(_synthetic_inchikey("CCCCCC")),
        synthesis_step=ReactionStep(reactants=[intermediate_left_2, intermediate_right]),
    )

    return Route(target=target, rank=1)


# =============================================================================
# Tests for get_stock_intermediates
# =============================================================================


@pytest.mark.unit
class TestGetStockIntermediates:
    """Tests for identifying which intermediate nodes are in stock."""

    def test_no_intermediates_in_stock(self):
        """Linear route where no intermediates are in stock."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Stock only contains the original leaf
        stock = {_synthetic_inchikey("C")}

        result = get_stock_intermediates(route, stock)

        assert len(result) == 0

    def test_single_intermediate_in_stock(self):
        """Linear route with one intermediate in stock."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Stock contains leaf + one intermediate (CC)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = get_stock_intermediates(route, stock)

        assert len(result) == 1
        intermediate = list(result)[0]
        assert intermediate.smiles == "CC"
        assert intermediate.inchikey == _synthetic_inchikey("CC")

    def test_all_intermediates_in_stock(self):
        """Linear route where all intermediates are in stock."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Route: CCCC <- CCC <- CC <- C
        # Intermediates: CCC, CC (not CCCC which is target)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
            _synthetic_inchikey("CCCC"),
        }

        result = get_stock_intermediates(route, stock)

        # Should find CC and CCC (not the target CCCC)
        assert len(result) == 2
        smiles_set = {mol.smiles for mol in result}
        assert smiles_set == {"CC", "CCC"}

    def test_leaf_route_has_no_intermediates(self):
        """Route with only a leaf has no intermediates."""
        leaf = _make_leaf_molecule("C")
        route = Route(target=leaf, rank=1)
        stock = {_synthetic_inchikey("C")}

        result = get_stock_intermediates(route, stock)

        assert len(result) == 0

    def test_single_step_route_has_no_intermediates(self):
        """Single-step route has no intermediates (only leaf and target)."""
        route = _make_linear_route_with_intermediates(depth=1)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = get_stock_intermediates(route, stock)

        # CC is the target, C is the leaf, no intermediates
        assert len(result) == 0

    def test_convergent_route_intermediates(self):
        """Convergent route with intermediates in stock."""
        route = _make_convergent_route_with_intermediates()
        # Add some intermediates to stock
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("O"),
            _synthetic_inchikey("branch_left_CC"),  # intermediate
            _synthetic_inchikey("branch_left_CCCC"),  # intermediate
        }

        result = get_stock_intermediates(route, stock)

        # Should find the two intermediates from left branch
        assert len(result) == 2
        smiles_set = {mol.smiles for mol in result}
        assert smiles_set == {"CC", "CCCC"}

    def test_excludes_target_molecule(self):
        """Target molecule should never be considered an intermediate."""
        route = _make_linear_route_with_intermediates(depth=2)
        # Put everything including target in stock
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),  # This is the target
        }

        result = get_stock_intermediates(route, stock)

        # Should only find CC, not CCC (the target)
        assert len(result) == 1
        assert list(result)[0].smiles == "CC"


# =============================================================================
# Tests for generate_pruned_routes
# =============================================================================


@pytest.mark.unit
class TestGeneratePrunedRoutes:
    """Tests for generating all possible pruned route variants."""

    def test_no_intermediates_returns_original_only(self):
        """If no intermediates are in stock, return only the original route."""
        route = _make_linear_route_with_intermediates(depth=3)
        stock = {_synthetic_inchikey("C")}  # Only the original leaf

        result = generate_pruned_routes(route, stock)

        # Should return just the original route
        assert len(result) == 1
        assert result[0].target.smiles == route.target.smiles
        assert result[0].length == route.length

    def test_single_intermediate_generates_two_routes(self):
        """One intermediate in stock should generate 2 routes (original + pruned)."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Route: CCCC <- CCC <- CC <- C
        # Make CC available in stock
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        # Should get:
        # 1. Original: CCCC <- CCC <- CC <- C (length 3)
        # 2. Pruned:   CCCC <- CCC <- CC (length 2, CC is now a leaf)
        assert len(result) == 2

        lengths = sorted([r.length for r in result])
        assert lengths == [2, 3]

        # Find the pruned route (length 2)
        pruned = [r for r in result if r.length == 2][0]
        # CC should be a leaf in the pruned route
        leaves = pruned.leaves
        leaf_smiles = {leaf.smiles for leaf in leaves}
        assert "CC" in leaf_smiles

    def test_two_intermediates_generate_three_routes_linear(self):
        """
        Two chained intermediates in a linear route generate 3 routes (not 4).

        In a linear route, pruning at an ancestor makes descendants unreachable,
        so the combination {ancestor, descendant} is redundant with {ancestor}.
        Only antichains (no ancestor-descendant pairs) are generated.
        """
        route = _make_linear_route_with_intermediates(depth=4)
        # Route: CCCCC <- CCCC <- CCC <- CC <- C
        # Intermediates: CCCC, CCC, CC (CCCC is target, not intermediate)
        # CC and CCC are in a chain: CCC is ancestor of CC
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        # Should get 3 routes (antichains):
        # 1. Original: {} (no pruning) - length 4
        # 2. {CC} (prune at CC only) - length 3
        # 3. {CCC} (prune at CCC only) - length 2
        # Note: {CC, CCC} is NOT generated (CCC is ancestor of CC)
        assert len(result) == 3

        lengths = sorted([r.length for r in result])
        assert lengths == [2, 3, 4]

    def test_preserves_route_metadata(self):
        """Pruned routes should preserve metadata from original."""
        route = _make_linear_route_with_intermediates(depth=2)
        route.metadata["test_key"] = "test_value"
        route.rank = 5

        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        for pruned_route in result:
            assert pruned_route.metadata.get("test_key") == "test_value"
            assert pruned_route.rank == 5

    def test_all_routes_are_valid(self):
        """All generated routes should be valid Route objects."""
        route = _make_linear_route_with_intermediates(depth=3)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        for pruned_route in result:
            # Should be a valid Route
            assert isinstance(pruned_route, Route)
            # Should have same target molecule
            assert pruned_route.target.smiles == route.target.smiles
            # Length should be computable
            assert pruned_route.length >= 0
            # Leaves should be computable
            assert len(pruned_route.leaves) > 0

    def test_leaf_route_returns_empty_list(self):
        """A leaf-only route should return empty list (no reactions to prune)."""
        leaf = _make_leaf_molecule("C")
        route = Route(target=leaf, rank=1)
        stock = {_synthetic_inchikey("C")}

        result = generate_pruned_routes(route, stock)

        # Leaf routes have no reactions, so nothing to generate
        assert len(result) == 0

    def test_single_step_route_with_stock_intermediate(self):
        """Single-step route has no intermediates to prune."""
        route = _make_linear_route_with_intermediates(depth=1)
        # Route: CC <- C
        # Even if we add CC to stock, it's the target, not an intermediate
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        # Should just return the original route
        assert len(result) == 1
        assert result[0].length == 1

    def test_convergent_route_pruning(self):
        """
        Convergent routes with intermediates in different branches can combine independently.

        Structure:
          Left branch: CCCC <- CC <- C (2 intermediates in chain)
          Right branch: CC <- O (1 intermediate)

        Left branch antichains: {}, {left_CC}, {left_CCCC} (3 options)
        Right branch antichains: {}, {right_CC} (2 options)
        Total: 3 * 2 = 6 routes
        """
        route = _make_convergent_route_with_intermediates()
        # Make all intermediates available
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("O"),
            _synthetic_inchikey("branch_left_CC"),
            _synthetic_inchikey("branch_left_CCCC"),
            _synthetic_inchikey("branch_right_CC"),
        }

        result = generate_pruned_routes(route, stock)

        # Should generate 6 routes (not 8):
        # Left branch has 2 chained intermediates -> 3 antichains
        # Right branch has 1 intermediate -> 2 antichains
        # Total: 3 * 2 = 6 (independent branches multiply)
        assert len(result) == 6

        # All should be valid
        for pruned_route in result:
            assert isinstance(pruned_route, Route)
            assert pruned_route.target.smiles == "CCCCCC"

    def test_pruned_routes_have_correct_leaves(self):
        """Pruned routes should have correct leaf sets."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Route: CCCC <- CCC <- CC <- C
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        # Original route: leaves = {C}
        original = [r for r in result if r.length == 3][0]
        assert len(original.leaves) == 1
        assert list(original.leaves)[0].smiles == "C"

        # Pruned route: leaves = {CC}
        pruned = [r for r in result if r.length == 2][0]
        assert len(pruned.leaves) == 1
        assert list(pruned.leaves)[0].smiles == "CC"

    def test_pruned_routes_are_solvable(self):
        """All pruned routes should be solvable with the stock."""
        route = _make_linear_route_with_intermediates(depth=3)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        # Every route should have all leaves in stock
        for pruned_route in result:
            for leaf in pruned_route.leaves:
                assert leaf.inchikey in stock, f"Leaf {leaf.smiles} not in stock"

    def test_original_route_always_included(self):
        """The original route should always be in the results."""
        route = _make_linear_route_with_intermediates(depth=3)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
        }

        result = generate_pruned_routes(route, stock)

        # One route should have the same signature as original
        original_sig = route.get_signature()
        result_sigs = [r.get_signature() for r in result]
        assert original_sig in result_sigs


# =============================================================================
# Property-based tests with hypothesis
# =============================================================================


@pytest.mark.unit
class TestGeneratePrunedRoutesProperties:
    """Property-based tests for route pruning invariants."""

    def test_all_routes_have_same_target(self):
        """Property: All pruned routes must have the same target molecule."""
        route = _make_linear_route_with_intermediates(depth=5)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        target_smiles = route.target.smiles
        for pruned_route in result:
            assert pruned_route.target.smiles == target_smiles
            assert pruned_route.target.inchikey == route.target.inchikey

    def test_pruned_routes_are_shorter_or_equal(self):
        """Property: Pruned routes should never be longer than original."""
        route = _make_linear_route_with_intermediates(depth=4)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        original_length = route.length
        for pruned_route in result:
            assert pruned_route.length <= original_length

    def test_num_routes_for_chained_intermediates(self):
        """
        Property: n chained intermediates in a linear route -> n+1 routes.

        In a linear route, intermediates form a chain where each is an ancestor
        of the next. Only antichains are valid, which are: {} and each singleton.
        So n chained intermediates generate n+1 antichains (empty + n singletons).
        """
        route = _make_linear_route_with_intermediates(depth=5)
        # Route: CCCCCC <- CCCCC <- CCCC <- CCC <- CC <- C
        # Intermediates: CCCCC, CCCC, CCC, CC (4 intermediates, but target is CCCCCC)
        # Put 2 of them in stock: CC and CCC (both in the chain)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        # 2 chained intermediates -> 2+1 = 3 routes: {}, {CC}, {CCC}
        assert len(result) == 3

    def test_all_routes_have_unique_signatures(self):
        """
        Property: All generated routes should have unique topologies.

        With antichain filtering, we guarantee no redundant pruning combinations,
        so every generated route should have a unique signature.
        """
        route = _make_linear_route_with_intermediates(depth=4)
        stock = {
            _synthetic_inchikey("C"),
            _synthetic_inchikey("CC"),
            _synthetic_inchikey("CCC"),
        }

        result = generate_pruned_routes(route, stock)

        signatures = [r.get_signature() for r in result]
        # All signatures should be unique (antichain filtering prevents duplicates)
        assert len(signatures) == len(set(signatures))

        # Should have exactly 3 unique routes for 2 chained intermediates
        assert len(signatures) == 3


# =============================================================================
# Property-based tests with hypothesis (for deeper invariant checking)
# =============================================================================


@pytest.mark.unit
class TestGeneratePrunedRoutesHypothesis:
    """Property-based tests using hypothesis for comprehensive edge case coverage."""

    def test_hypothesis_all_pruned_routes_solvable(self):
        """
        Property: Every generated route must be solvable with the stock.

        Uses hypothesis to test with various route depths and stock configurations.
        """

        @given(
            depth=st.integers(min_value=1, max_value=8), num_stock_intermediates=st.integers(min_value=0, max_value=5)
        )
        def check_all_solvable(depth, num_stock_intermediates):
            route = _make_linear_route_with_intermediates(depth=depth)

            # Create stock with original leaf + some intermediates
            stock = {_synthetic_inchikey("C")}

            # Add random intermediates to stock
            for i in range(2, min(depth + 1, num_stock_intermediates + 2)):
                stock.add(_synthetic_inchikey("C" * i))

            result = generate_pruned_routes(route, stock)

            for pruned_route in result:
                assert is_route_solved(pruned_route, stock), (
                    f"Generated unsolvable route with length {pruned_route.length}"
                )

        check_all_solvable()

    def test_hypothesis_pruned_never_longer(self):
        """Property: Pruned routes should never be longer than original."""

        @given(depth=st.integers(min_value=2, max_value=8))
        def check_never_longer(depth):
            route = _make_linear_route_with_intermediates(depth=depth)

            # Add all intermediates to stock
            stock = {_synthetic_inchikey("C" * i) for i in range(1, depth + 2)}

            result = generate_pruned_routes(route, stock)

            original_length = route.length
            for pruned_route in result:
                assert pruned_route.length <= original_length, (
                    f"Pruned route longer than original: {pruned_route.length} > {original_length}"
                )

        check_never_longer()

    def test_hypothesis_same_target_molecule(self):
        """Property: All pruned routes must have identical target molecule."""

        @given(depth=st.integers(min_value=1, max_value=8))
        def check_same_target(depth):
            route = _make_linear_route_with_intermediates(depth=depth)

            stock = {_synthetic_inchikey("C" * i) for i in range(1, depth + 2)}

            result = generate_pruned_routes(route, stock)

            original_target_inchikey = route.target.inchikey
            for pruned_route in result:
                assert pruned_route.target.inchikey == original_target_inchikey
                assert pruned_route.target.smiles == route.target.smiles

        check_same_target()

    def test_hypothesis_antichain_property(self):
        """
        Property: For linear routes, number of routes = number of stock intermediates + 1.

        This validates that antichain filtering works correctly for all chain lengths.
        """

        @given(depth=st.integers(min_value=1, max_value=8))
        def check_antichain_count(depth):
            route = _make_linear_route_with_intermediates(depth=depth)

            # Add all intermediates to stock (they form a chain)
            stock = {_synthetic_inchikey("C" * i) for i in range(1, depth + 2)}

            # Count how many are actually intermediates (not target, not original leaf)
            stock_ints = get_stock_intermediates(route, stock)
            num_intermediates = len(stock_ints)

            result = generate_pruned_routes(route, stock)

            # For a chain: n intermediates → n+1 antichains (empty set + n singletons)
            expected = num_intermediates + 1
            assert len(result) == expected, (
                f"Chain with {num_intermediates} intermediates should generate {expected} routes, got {len(result)}"
            )

        check_antichain_count()

    def test_hypothesis_unique_signatures(self):
        """Property: All generated routes must have unique signatures (no duplicates)."""

        @given(depth=st.integers(min_value=2, max_value=7))
        def check_unique_sigs(depth):
            route = _make_linear_route_with_intermediates(depth=depth)

            stock = {_synthetic_inchikey("C" * i) for i in range(1, depth + 2)}

            result = generate_pruned_routes(route, stock)

            signatures = [r.get_signature() for r in result]
            assert len(signatures) == len(set(signatures)), f"Found duplicate signatures in {len(result)} routes"

        check_unique_sigs()

    def test_hypothesis_original_included_when_solvable(self):
        """Property: Original route is always included if it's solvable."""

        @given(depth=st.integers(min_value=1, max_value=8))
        def check_original_included(depth):
            route = _make_linear_route_with_intermediates(depth=depth)

            # Make stock contain all leaves
            stock = {_synthetic_inchikey("C" * i) for i in range(1, depth + 2)}

            result = generate_pruned_routes(route, stock)

            # Original signature should be in results
            original_sig = route.get_signature()
            result_sigs = [r.get_signature() for r in result]

            assert original_sig in result_sigs, "Original solvable route not included in results"

        check_original_included()


# =============================================================================
# Edge cases and error conditions
# =============================================================================


@pytest.mark.unit
class TestGeneratePrunedRoutesEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_stock(self):
        """Empty stock should raise or return empty (depends on implementation choice)."""
        route = _make_linear_route_with_intermediates(depth=2)
        stock = set()  # Empty stock

        result = generate_pruned_routes(route, stock)

        # With empty stock, no leaves are satisfied, should return empty
        assert len(result) == 0

    def test_stock_with_only_intermediates(self):
        """Stock containing only intermediates (no original leaves)."""
        route = _make_linear_route_with_intermediates(depth=3)
        # Route: CCCC <- CCC <- CC <- C
        # Stock has CC but not C (the original leaf)
        stock = {
            _synthetic_inchikey("CC"),  # intermediate
        }

        result = generate_pruned_routes(route, stock)

        # Original route is not solvable (C not in stock)
        # But pruned route starting at CC is solvable
        # Should generate 1 route (pruned to CC)
        assert len(result) == 1
        assert result[0].length == 2

    def test_max_intermediates(self):
        """
        Route with maximum realistic intermediates (depth 10).

        In a linear route, all intermediates form a single chain, so we get
        n+1 routes instead of 2^n. With 9 chained intermediates, we get 10 routes.
        """
        route = _make_linear_route_with_intermediates(depth=10)
        # Add all intermediates to stock
        stock = {_synthetic_inchikey("C" * i) for i in range(1, 12)}

        result = generate_pruned_routes(route, stock)

        # 9 intermediates in a linear chain (depth 10 has intermediates at positions 2-10)
        # With antichain filtering: 9+1 = 10 routes (not 2^9 = 512)
        # This is the efficiency gain of pre-filtering!
        assert len(result) == 10
