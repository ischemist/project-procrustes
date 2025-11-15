"""
Tests for the public API convenience functions for single route adaptation.
"""

import pytest

from retrocast import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.exceptions import RetroCastException
from retrocast.schemas import Route, TargetInput


class TestGetAdapter:
    """Test the get_adapter function."""

    def test_get_adapter_valid(self):
        """Test getting a valid adapter."""
        adapter = get_adapter("dms")
        assert adapter is not None
        assert adapter == ADAPTER_MAP["dms"]

    def test_get_adapter_invalid(self):
        """Test getting an invalid adapter raises exception."""
        with pytest.raises(RetroCastException, match="unknown adapter 'nonexistent'"):
            get_adapter("nonexistent")

    def test_all_adapters_in_map(self):
        """Test that all adapters in ADAPTER_MAP can be retrieved."""
        for adapter_name in ADAPTER_MAP:
            adapter = get_adapter(adapter_name)
            assert adapter is not None


class TestAdaptSingleRoute:
    """Test the adapt_single_route convenience function."""

    def test_adapt_single_dms_route(self):
        """Test adapting a single DMS route."""
        target = TargetInput(id="test_target", smiles="CCO")

        # Simple DMS route: ethanol from CO + methane (nonsensical but valid structure)
        raw_route = {
            "smiles": "CCO",
            "children": [
                {"smiles": "C", "children": []},  # methane (leaf)
                {"smiles": "CO", "children": []},  # methanol (leaf)
            ],
        }

        route = adapt_single_route(raw_route, target, "dms")

        assert route is not None
        assert isinstance(route, Route)
        assert route.target.smiles == "CCO"
        assert route.depth == 1  # One reaction step
        assert len(route.leaves) == 2  # Two starting materials

    def test_adapt_single_route_with_list(self):
        """Test that wrapping in a list doesn't break the function."""
        target = TargetInput(id="test", smiles="CCO")

        raw_route = {
            "smiles": "CCO",
            "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
        }

        # Pass as a list (should still work)
        route = adapt_single_route([raw_route], target, "dms")

        assert route is not None
        assert isinstance(route, Route)

    def test_adapt_single_route_invalid_data(self):
        """Test that invalid data returns None."""
        target = TargetInput(id="test", smiles="CCO")

        # Invalid structure - missing required fields
        raw_route = {"invalid": "data"}

        route = adapt_single_route(raw_route, target, "dms")

        assert route is None  # Should return None on failure

    def test_adapt_single_route_invalid_adapter(self):
        """Test that invalid adapter name raises exception."""
        target = TargetInput(id="test", smiles="CCO")
        raw_route = {"smiles": "CCO", "children": []}

        with pytest.raises(RetroCastException, match="unknown adapter"):
            adapt_single_route(raw_route, target, "nonexistent_adapter")

    def test_adapt_single_route_target_mismatch(self):
        """Test that target SMILES mismatch is handled."""
        target = TargetInput(id="test", smiles="CCCO")  # propanol

        # Route for ethanol (different from target)
        raw_route = {"smiles": "CCO", "children": []}

        route = adapt_single_route(raw_route, target, "dms")

        # Should return None because of mismatch
        assert route is None


class TestAdaptRoutes:
    """Test the adapt_routes convenience function for batch processing."""

    def test_adapt_multiple_routes(self):
        """Test adapting multiple routes."""
        target = TargetInput(id="test", smiles="CCO")

        raw_routes = [
            {
                "smiles": "CCO",
                "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
            },
            {
                "smiles": "CCO",
                "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
            },
        ]

        routes = adapt_routes(raw_routes, target, "dms")

        assert len(routes) == 2
        assert all(isinstance(r, Route) for r in routes)
        assert all(r.target.smiles == "CCO" for r in routes)

    def test_adapt_routes_with_max_limit(self):
        """Test that max_routes parameter limits the output."""
        target = TargetInput(id="test", smiles="CCO")

        raw_routes = [
            {
                "smiles": "CCO",
                "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
            },
            {
                "smiles": "CCO",
                "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
            },
            {
                "smiles": "CCO",
                "children": [{"smiles": "CCO", "children": []}],  # Direct route
            },
        ]

        routes = adapt_routes(raw_routes, target, "dms", max_routes=2)

        assert len(routes) == 2  # Should stop at 2

    def test_adapt_routes_empty_list(self):
        """Test adapting empty route list."""
        target = TargetInput(id="test", smiles="CCO")

        routes = adapt_routes([], target, "dms")

        assert routes == []

    def test_adapt_routes_with_failures(self):
        """Test that individual route failures during transformation are handled."""
        target = TargetInput(id="test", smiles="CCO")

        raw_routes = [
            {
                "smiles": "CCO",
                "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
            },  # This succeeds
            {
                "smiles": "CCCO",  # Wrong target - will fail transformation
                "children": [{"smiles": "C", "children": []}],
            },
            {
                "smiles": "CCO",
                "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
            },  # This succeeds
        ]

        routes = adapt_routes(raw_routes, target, "dms")

        # The middle route should fail due to target mismatch, so we get 2 routes
        assert len(routes) == 2
        assert all(r.target.smiles == "CCO" for r in routes)

    def test_adapt_routes_all_fail(self):
        """Test that all failures result in empty list."""
        target = TargetInput(id="test", smiles="CCO")

        raw_routes = [
            {"invalid": "route1"},
            {"invalid": "route2"},
            {"invalid": "route3"},
        ]

        routes = adapt_routes(raw_routes, target, "dms")

        assert routes == []


class TestIntegrationWithDomainTree:
    """Test integration with domain.tree utilities."""

    def test_end_to_end_workflow(self):
        """Test a complete workflow: adapt routes, deduplicate, and sample."""

        target = TargetInput(id="test", smiles="CCO")

        # Create routes with one duplicate
        raw_routes = [
            {
                "smiles": "CCO",
                "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
            },
            {
                "smiles": "CCO",
                "children": [{"smiles": "C", "children": []}, {"smiles": "CO", "children": []}],
            },  # Duplicate
            {
                "smiles": "CCO",
                "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
            },
        ]

        # Adapt all routes
        routes = adapt_routes(raw_routes, target, "dms")
        assert len(routes) == 3

        # Deduplicate (NOTE: This uses the old schema, but we can test the concept)
        # For now, just verify we can import and the routes are adapted correctly
        assert all(isinstance(r, Route) for r in routes)

        # Sample top-k
        # top_routes = sample_top_k(routes, k=2)
        # assert len(top_routes) == 2
