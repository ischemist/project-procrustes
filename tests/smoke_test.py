"""
Smoke test to verify package installation and basic functionality.

This test is run as part of the PyPI publishing workflow to ensure
the built package works correctly before publishing.

Run with: pytest tests/smoke_test.py
"""

import retrocast


def test_version_exists():
    """Verify version is accessible and valid."""
    assert hasattr(retrocast, "__version__")
    assert isinstance(retrocast.__version__, str)
    assert len(retrocast.__version__) > 0
    # Should contain at least major.minor
    parts = retrocast.__version__.split(".")
    assert len(parts) >= 2, f"Version '{retrocast.__version__}' should have at least major.minor"


def test_core_exports():
    """Verify all main exports are available."""
    # Core schemas
    assert hasattr(retrocast, "Route")
    assert hasattr(retrocast, "Molecule")
    assert hasattr(retrocast, "ReactionStep")
    assert hasattr(retrocast, "TargetInput")

    # Adapter functions
    assert hasattr(retrocast, "adapt_single_route")
    assert hasattr(retrocast, "adapt_routes")
    assert hasattr(retrocast, "get_adapter")
    assert hasattr(retrocast, "ADAPTER_MAP")

    # Route processing utilities
    assert hasattr(retrocast, "deduplicate_routes")
    assert hasattr(retrocast, "sample_top_k")
    assert hasattr(retrocast, "sample_random_k")
    assert hasattr(retrocast, "sample_k_by_depth")


def test_basic_adaptation():
    """Verify basic adapter functionality works."""
    from retrocast import Route, TargetInput, adapt_single_route

    target = TargetInput(id="test", smiles="CCO")
    raw_route = {
        "smiles": "CCO",
        "children": [
            {"smiles": "C", "children": []},
            {"smiles": "CO", "children": []},
        ],
    }

    route = adapt_single_route(raw_route, target, "dms")

    assert route is not None
    assert isinstance(route, Route)
    assert route.target.smiles == "CCO"
    assert route.depth == 1
    assert len(route.leaves) == 2


def test_route_has_version():
    """Verify routes include retrocast version for provenance."""
    from retrocast import TargetInput, adapt_single_route

    target = TargetInput(id="test", smiles="CCO")
    raw_route = {
        "smiles": "CCO",
        "children": [
            {"smiles": "C", "children": []},
            {"smiles": "CO", "children": []},
        ],
    }

    route = adapt_single_route(raw_route, target, "dms")

    assert route is not None
    assert hasattr(route, "retrocast_version")
    assert isinstance(route.retrocast_version, str)
    assert len(route.retrocast_version) > 0
    # Version should match package version
    assert route.retrocast_version == retrocast.__version__


def test_adapter_map_not_empty():
    """Verify adapter map contains registered adapters."""
    from retrocast import ADAPTER_MAP

    assert len(ADAPTER_MAP) > 0
    # Check a few known adapters exist
    assert "dms" in ADAPTER_MAP
    assert "synplanner" in ADAPTER_MAP
