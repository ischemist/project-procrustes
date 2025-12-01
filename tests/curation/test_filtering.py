"""
Unit tests for retrocast.curation.filtering module.

Tests filtering functions using synthetic carbon-chain routes and
real-ish molecule fixtures.
"""

import pytest

from retrocast.curation.filtering import (
    clean_and_prioritize_pools,
    deduplicate_routes,
    excise_reactions_from_route,
    filter_by_route_type,
)
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr
from tests.helpers import _synthetic_inchikey

# =============================================================================
# Helper functions for creating test data
# =============================================================================


def make_leaf(smiles: str, inchikey: str) -> Molecule:
    """Create a leaf molecule."""
    return Molecule(
        smiles=SmilesStr(smiles),
        inchikey=InchiKeyStr(inchikey),
        synthesis_step=None,
    )


def make_product(smiles: str, inchikey: str, reactants: list[Molecule]) -> Molecule:
    """Create a product molecule with synthesis step."""
    return Molecule(
        smiles=SmilesStr(smiles),
        inchikey=InchiKeyStr(inchikey),
        synthesis_step=ReactionStep(reactants=reactants),
    )


# =============================================================================
# Tests for excise_reactions_from_route
# =============================================================================


@pytest.mark.unit
class TestExciseReactionsFromRoute:
    def test_excise_from_linear_route(self):
        """Excising middle reaction should split into two routes."""
        # Build route: D <- C <- B <- A
        # D is target, A is leaf
        leaf_a = make_leaf("A", "INCHI-A")
        mol_b = make_product("B", "INCHI-B", [leaf_a])
        mol_c = make_product("C", "INCHI-C", [mol_b])
        mol_d = make_product("D", "INCHI-D", [mol_c])

        route = Route(target=mol_d, rank=1)

        # Excise C <- B (the reaction producing C from B)
        sig_c_from_b: ReactionSignature = (frozenset(["INCHI-B"]), "INCHI-C")

        result = excise_reactions_from_route(route, {sig_c_from_b})

        # Should get:
        # 1. Main route: D <- C (where C is now a leaf)
        # 2. Sub-route: B <- A
        assert len(result) == 2

        # Main route
        main = result[0]
        assert main.target.inchikey == "INCHI-D"
        assert main.target.synthesis_step is not None
        # C should now be a leaf in the main route
        c_node = main.target.synthesis_step.reactants[0]
        assert c_node.inchikey == "INCHI-C"
        assert c_node.is_leaf

        # Sub-route
        sub = result[1]
        assert sub.target.inchikey == "INCHI-B"
        assert sub.target.synthesis_step is not None
        assert sub.target.synthesis_step.reactants[0].inchikey == "INCHI-A"

    def test_excise_first_reaction(self):
        """Excising first (deepest) reaction should make its product a leaf."""
        # Route: C <- B <- A
        leaf_a = make_leaf("A", "INCHI-A")
        mol_b = make_product("B", "INCHI-B", [leaf_a])
        mol_c = make_product("C", "INCHI-C", [mol_b])

        route = Route(target=mol_c, rank=1)

        # Excise B <- A
        sig_b_from_a: ReactionSignature = (frozenset(["INCHI-A"]), "INCHI-B")

        result = excise_reactions_from_route(route, {sig_b_from_a})

        # Should get just main route with B as leaf
        assert len(result) == 1
        main = result[0]
        assert main.target.inchikey == "INCHI-C"
        b_node = main.target.synthesis_step.reactants[0]
        assert b_node.inchikey == "INCHI-B"
        assert b_node.is_leaf

    def test_excise_last_reaction(self):
        """Excising the last (top) reaction should return empty or sub-routes."""
        # Route: C <- B <- A
        leaf_a = make_leaf("A", "INCHI-A")
        mol_b = make_product("B", "INCHI-B", [leaf_a])
        mol_c = make_product("C", "INCHI-C", [mol_b])

        route = Route(target=mol_c, rank=1)

        # Excise C <- B (the top reaction)
        sig_c_from_b: ReactionSignature = (frozenset(["INCHI-B"]), "INCHI-C")

        result = excise_reactions_from_route(route, {sig_c_from_b})

        # Main route has no reactions, so only sub-route: B <- A
        assert len(result) == 1
        sub = result[0]
        assert sub.target.inchikey == "INCHI-B"

    def test_excise_from_convergent_route(self):
        """Excising from convergent route should handle branches correctly."""
        # Route: D <- (B, C)
        #        B <- A
        #        C is leaf
        leaf_a = make_leaf("A", "INCHI-A")
        leaf_c = make_leaf("C", "INCHI-C")
        mol_b = make_product("B", "INCHI-B", [leaf_a])
        mol_d = make_product("D", "INCHI-D", [mol_b, leaf_c])

        route = Route(target=mol_d, rank=1)

        # Excise the top reaction D <- (B, C)
        sig_d: ReactionSignature = (frozenset(["INCHI-B", "INCHI-C"]), "INCHI-D")

        result = excise_reactions_from_route(route, {sig_d})

        # Main route has no reactions
        # Sub-route from B (which has reactions): B <- A
        # C is leaf, no sub-route
        assert len(result) == 1
        assert result[0].target.inchikey == "INCHI-B"

    def test_excise_nothing(self):
        """Excising with empty set should return full route."""
        leaf_a = make_leaf("A", "INCHI-A")
        mol_b = make_product("B", "INCHI-B", [leaf_a])

        route = Route(target=mol_b, rank=1)

        result = excise_reactions_from_route(route, set())

        assert len(result) == 1
        assert result[0].target.inchikey == "INCHI-B"
        assert result[0].target.synthesis_step is not None

    def test_excise_from_leaf_route(self):
        """Route with only a leaf (no reactions) should return empty."""
        leaf = make_leaf("A", "INCHI-A")
        route = Route(target=leaf, rank=1)

        result = excise_reactions_from_route(route, set())

        assert len(result) == 0

    def test_preserves_metadata(self):
        """Excision should preserve route metadata."""
        leaf_a = make_leaf("A", "INCHI-A")
        mol_b = make_product("B", "INCHI-B", [leaf_a])
        mol_c = make_product("C", "INCHI-C", [mol_b])

        route = Route(
            target=mol_c,
            rank=1,
            metadata={"source": "test"},
        )

        sig_c_from_b: ReactionSignature = (frozenset(["INCHI-B"]), "INCHI-C")

        result = excise_reactions_from_route(route, {sig_c_from_b})

        # Sub-route should have same rank
        sub = result[0]
        assert sub.rank == 1


# =============================================================================
# Tests for deduplicate_routes
# =============================================================================


@pytest.mark.unit
class TestDeduplicateRoutes:
    def test_removes_duplicate_routes(self, synthetic_route_factory):
        """Should remove routes with identical signatures."""
        route1 = synthetic_route_factory("linear", depth=2)
        route2 = synthetic_route_factory("linear", depth=2)  # Same structure

        # They have same signature
        assert route1.get_signature() == route2.get_signature()

        result = deduplicate_routes([route1, route2])

        assert len(result) == 1
        assert result[0] is route1  # First one kept

    def test_keeps_different_routes(self, synthetic_route_factory):
        """Should keep routes with different signatures."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=2)

        # Different depths = different signatures
        assert route1.get_signature() != route2.get_signature()

        result = deduplicate_routes([route1, route2])

        assert len(result) == 2

    def test_preserves_order(self, synthetic_route_factory):
        """Should preserve original order of first occurrences."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=2)
        route3 = synthetic_route_factory("linear", depth=3)

        result = deduplicate_routes([route1, route2, route3])

        assert result[0] is route1
        assert result[1] is route2
        assert result[2] is route3

    def test_empty_list(self):
        """Should handle empty input."""
        result = deduplicate_routes([])
        assert result == []

    def test_single_route(self, synthetic_route_factory):
        """Should handle single route."""
        route = synthetic_route_factory("linear", depth=1)
        result = deduplicate_routes([route])
        assert len(result) == 1


# =============================================================================
# Tests for filter_by_route_type
# =============================================================================


@pytest.mark.unit
class TestFilterByRouteType:
    @pytest.fixture
    def benchmark_with_mixed_routes(self, synthetic_route_factory):
        """Create benchmark with linear and convergent routes."""
        linear_route = synthetic_route_factory("linear", depth=2)
        convergent_route = synthetic_route_factory("convergent", depth=2)

        linear_target = BenchmarkTarget(
            id="LINEAR-TARGET",
            smiles="CCC",
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[linear_route],
        )
        convergent_target = BenchmarkTarget(
            id="CONVERGENT-TARGET",
            smiles="CCCC",
            inchi_key=_synthetic_inchikey("CCCC"),
            acceptable_routes=[convergent_route],
        )

        return BenchmarkSet(
            name="test",
            targets={
                "LINEAR-TARGET": linear_target,
                "CONVERGENT-TARGET": convergent_target,
            },
        )

    def test_filter_linear_routes(self, benchmark_with_mixed_routes):
        """Should return only linear routes."""
        result = filter_by_route_type(benchmark_with_mixed_routes, "linear")

        assert len(result) == 1
        assert result[0].id == "LINEAR-TARGET"
        assert not result[0].is_convergent

    def test_filter_convergent_routes(self, benchmark_with_mixed_routes):
        """Should return only convergent routes."""
        result = filter_by_route_type(benchmark_with_mixed_routes, "convergent")

        assert len(result) == 1
        assert result[0].id == "CONVERGENT-TARGET"
        assert result[0].is_convergent

    def test_raises_on_unknown_type(self, benchmark_with_mixed_routes):
        """Should raise ValueError for unknown route type."""
        with pytest.raises(ValueError, match="Unknown route type"):
            filter_by_route_type(benchmark_with_mixed_routes, "invalid")  # type: ignore


# =============================================================================
# Tests for clean_and_prioritize_pools
# =============================================================================


@pytest.mark.unit
class TestCleanAndPrioritizePools:
    def test_removes_duplicate_route_signatures(self, synthetic_route_factory):
        """Should remove secondary targets with same route signature as primary."""
        route = synthetic_route_factory("linear", depth=2)

        primary_target = BenchmarkTarget(
            id="PRIMARY",
            smiles="CCC",
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[route],
        )
        # Same route signature in secondary
        secondary_target = BenchmarkTarget(
            id="SECONDARY",
            smiles="CCC-alt",
            inchi_key=_synthetic_inchikey("CCC-alt"),
            acceptable_routes=[route],
        )

        clean_p, clean_s = clean_and_prioritize_pools([primary_target], [secondary_target])

        assert len(clean_p) == 1
        assert len(clean_s) == 0

    def test_removes_ambiguous_smiles(self, synthetic_route_factory):
        """Should remove targets with same SMILES from both pools."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=2)

        # Same SMILES but different routes
        primary_target = BenchmarkTarget(
            id="PRIMARY",
            smiles="CCC",  # Same SMILES
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[route1],
        )
        secondary_target = BenchmarkTarget(
            id="SECONDARY",
            smiles="CCC",  # Same SMILES
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[route2],
        )

        clean_p, clean_s = clean_and_prioritize_pools([primary_target], [secondary_target])

        # Both should be removed due to ambiguity
        assert len(clean_p) == 0
        assert len(clean_s) == 0

    def test_keeps_non_conflicting_targets(self, synthetic_route_factory):
        """Should keep targets with no conflicts."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=2)

        primary_target = BenchmarkTarget(
            id="PRIMARY",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[route1],
        )
        secondary_target = BenchmarkTarget(
            id="SECONDARY",
            smiles="CCC",
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[route2],
        )

        clean_p, clean_s = clean_and_prioritize_pools([primary_target], [secondary_target])

        assert len(clean_p) == 1
        assert len(clean_s) == 1

    def test_empty_pools(self):
        """Should handle empty pools."""
        clean_p, clean_s = clean_and_prioritize_pools([], [])
        assert clean_p == []
        assert clean_s == []

    def test_targets_without_ground_truth(self):
        """Should handle targets without ground truth routes."""
        primary_target = BenchmarkTarget(
            id="PRIMARY",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
        )
        secondary_target = BenchmarkTarget(
            id="SECONDARY",
            smiles="CCC",
            inchi_key=_synthetic_inchikey("CCC"),
            acceptable_routes=[],
        )

        clean_p, clean_s = clean_and_prioritize_pools([primary_target], [secondary_target])

        # Both should be kept (no route signature to conflict)
        assert len(clean_p) == 1
        assert len(clean_s) == 1


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.integration
class TestFilteringIntegration:
    def test_deduplicate_then_filter_by_type(self, synthetic_route_factory):
        """Chain deduplication with type filtering."""
        # Create duplicate routes
        linear1 = synthetic_route_factory("linear", depth=2)
        linear2 = synthetic_route_factory("linear", depth=2)  # Duplicate
        convergent = synthetic_route_factory("convergent", depth=2)

        routes = [linear1, linear2, convergent]

        # First deduplicate
        unique = deduplicate_routes(routes)
        assert len(unique) == 2

        # Create benchmark for filtering
        targets = {}
        for i, route in enumerate(unique):
            smiles = f"SMILES_{i}"
            target = BenchmarkTarget(
                id=f"INCHI_{i}",
                smiles=smiles,
                inchi_key=_synthetic_inchikey(smiles),
                acceptable_routes=[route],
            )
            targets[f"INCHI_{i}"] = target

        benchmark = BenchmarkSet(name="test", targets=targets)

        # Filter by type
        linear_targets = filter_by_route_type(benchmark, "linear")
        convergent_targets = filter_by_route_type(benchmark, "convergent")

        assert len(linear_targets) == 1
        assert len(convergent_targets) == 1
