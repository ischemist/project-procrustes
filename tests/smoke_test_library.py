"""
Smoke tests for the retrocast library API.

These tests validate that the public API documented in docs/library.md
works correctly using real data files (DMS explorer XL predictions on mkt-cnv-160).

This ensures the documentation examples are accurate and functional.

Run with: pytest tests/smoke_test_library.py -v
"""

from pathlib import Path

import pytest

# Real data paths (used by integration tests)
DATA_DIR = Path("data")
BENCHMARK_PATH = DATA_DIR / "1-benchmarks/definitions/mkt-cnv-160.json.gz"
STOCK_PATH = DATA_DIR / "1-benchmarks/stocks/buyables-stock.txt"
ROUTES_PATH = DATA_DIR / "3-processed/mkt-cnv-160/dms-explorer-xl/routes.json.gz"


@pytest.mark.skipif(not BENCHMARK_PATH.exists(), reason="Real data files not available")
class TestLibraryAPIWithRealData:
    """Test library API examples from docs/library.md using real DMS data."""

    def test_adapt_single_route_example(self):
        """Test Section 1.1: Adapting a Single Route"""
        from retrocast import TargetInput, adapt_single_route

        # 1. Define the target context
        target = TargetInput(id="mol-1", smiles="CCO")

        # 2. Provide raw data (simple DMS-style tree structure)
        raw_data = {
            "smiles": "CCO",
            "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
        }

        # 3. Cast to Route
        route = adapt_single_route(raw_data, target, adapter_name="dms")

        # Validate the route was created successfully
        assert route is not None
        assert route.target.smiles == "CCO"
        assert route.length > 0
        assert len(route.leaves) == 2
        leaf_smiles = {m.smiles for m in route.leaves}
        assert "CC" in leaf_smiles
        assert "O" in leaf_smiles

    def test_adapt_routes_batch_example(self):
        """Test Section 1.2: Adapting Batch Predictions"""
        from retrocast import TargetInput, adapt_routes, deduplicate_routes

        # Simulate batch processing
        smiles_list = ["CCO", "CC(C)O"]
        targets = [TargetInput(id=f"t{i}", smiles=s) for i, s in enumerate(smiles_list)]

        # Mock model outputs (DMS-style)
        model_outputs = [
            # Two routes for CCO
            [
                {"smiles": "CCO", "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}]},
                {
                    "smiles": "CCO",
                    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
                },  # duplicate
            ],
            # One route for CC(C)O
            [{"smiles": "CC(C)O", "children": [{"smiles": "CC", "children": []}, {"smiles": "CO", "children": []}]}],
        ]

        all_routes = []
        for target, raw_output in zip(targets, model_outputs, strict=True):
            # Adapt
            routes = adapt_routes(raw_output, target, adapter_name="dms")

            # Deduplicate based on topological signature
            unique_routes = deduplicate_routes(routes)

            all_routes.extend(unique_routes)

        # After deduplication, should have 2 unique routes total
        assert len(all_routes) == 2
        assert all(route is not None for route in all_routes)

    def test_score_predictions_example(self):
        """Test Section 2.A: Score Predictions"""
        from retrocast.api import load_benchmark, load_routes, load_stock_file, score_predictions

        # 1. Load Resources
        benchmark = load_benchmark(BENCHMARK_PATH)
        stock = load_stock_file(STOCK_PATH)
        routes_dict = load_routes(ROUTES_PATH)

        # 2. Prepare Predictions (take first 5 targets for speed)
        target_ids = list(routes_dict.keys())[:5]
        predictions = {tid: routes_dict[tid] for tid in target_ids}

        # 3. Run Scoring
        results = score_predictions(
            benchmark=benchmark, predictions=predictions, stock=stock, model_name="DMS-Explorer-XL-Test"
        )

        # Validate results
        assert results is not None
        assert results.model_name == "DMS-Explorer-XL-Test"
        # Note: results include all benchmark targets, not just those with predictions
        assert len(results.results) >= len(predictions)

        # Access granular results for first target
        first_target_id = target_ids[0]
        t1_eval = results.results[first_target_id]
        assert hasattr(t1_eval, "is_solvable")
        assert hasattr(t1_eval, "acceptable_rank")
        # Verify the target we provided predictions for has routes
        assert len(t1_eval.routes) > 0

    def test_compute_statistics_example(self):
        """Test Section 2.B: Compute Statistics"""
        from retrocast.api import (
            compute_model_statistics,
            load_benchmark,
            load_routes,
            load_stock_file,
            score_predictions,
        )

        # Load and score (using small subset for speed)
        benchmark = load_benchmark(BENCHMARK_PATH)
        stock = load_stock_file(STOCK_PATH)
        routes_dict = load_routes(ROUTES_PATH)

        # Use first 10 targets
        predictions = {tid: routes_dict[tid] for tid in list(routes_dict.keys())[:10]}

        results = score_predictions(benchmark=benchmark, predictions=predictions, stock=stock, model_name="DMS-Test")

        # Compute stats from the scored results (fewer bootstrap iterations for speed)
        stats = compute_model_statistics(results, n_boot=100, seed=42)

        # Access aggregated metrics
        solvability = stats.solvability.overall
        assert solvability is not None
        assert 0.0 <= solvability.value <= 1.0
        assert hasattr(solvability, "ci_lower")
        assert hasattr(solvability, "ci_upper")

        # Access stratified metrics (e.g., by route length)
        assert hasattr(stats.solvability, "by_group")
        assert len(stats.solvability.by_group) > 0

    def test_adapter_map_reference_example(self):
        """Test Reference: Available Adapters"""
        from retrocast import ADAPTER_MAP

        # Check that ADAPTER_MAP is accessible
        adapter_names = list(ADAPTER_MAP.keys())
        assert len(adapter_names) > 0

        # Verify expected adapters are present (from docs)
        expected_adapters = ["aizynth", "dms", "retrostar", "askcos"]
        for adapter in expected_adapters:
            assert adapter in adapter_names, f"Expected adapter '{adapter}' not found in ADAPTER_MAP"


class TestLibraryAPIMinimal:
    """Minimal tests that don't require external data files."""

    def test_import_library_api(self):
        """Test that all documented API functions can be imported."""
        from retrocast import (
            ADAPTER_MAP,
            TargetInput,
            adapt_routes,
            adapt_single_route,
            deduplicate_routes,
        )
        from retrocast.api import (
            compute_model_statistics,
            load_benchmark,
            load_routes,
            load_stock_file,
            score_predictions,
        )

        # Verify functions are callable
        assert callable(adapt_single_route)
        assert callable(adapt_routes)
        assert callable(deduplicate_routes)
        assert callable(load_benchmark)
        assert callable(load_routes)
        assert callable(load_stock_file)
        assert callable(score_predictions)
        assert callable(compute_model_statistics)

        # Verify classes
        assert TargetInput is not None
        assert isinstance(ADAPTER_MAP, dict)

    def test_target_input_creation(self):
        """Test creating TargetInput objects as shown in docs."""
        from retrocast import TargetInput

        # Example from docs
        target = TargetInput(id="mol-1", smiles="CCO")
        assert target.id == "mol-1"
        assert target.smiles == "CCO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
