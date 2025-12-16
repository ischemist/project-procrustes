"""
Integration tests for the retrocast workflow pipeline.

Tests the full ingest -> score -> stats flow using synthetic data and tmp_path.
No mocking - uses real file I/O and synthetic carbon chain routes.
"""

from collections.abc import Generator
from typing import Any

import pytest

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.io.data import load_routes
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import InchiKeyStr, SmilesStr
from retrocast.workflow.ingest import ingest_model_predictions
from retrocast.workflow.score import score_model
from tests.helpers import _make_simple_route, _make_two_step_route, _synthetic_inchikey

# =============================================================================
# Test Adapter - Minimal adapter for synthetic data
# =============================================================================


class SyntheticAdapter(BaseAdapter):
    """
    Minimal adapter that passes through Route objects directly.

    Expects raw_target_data to be a list of Route dicts or Route objects.
    """

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        if not isinstance(raw_target_data, list):
            return

        for item in raw_target_data:
            if isinstance(item, Route):
                yield item
            elif isinstance(item, dict):
                # Parse Route from dict
                try:
                    route = Route.model_validate(item)
                    yield route
                except Exception:
                    continue


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_benchmark() -> BenchmarkSet:
    """
    Create a minimal benchmark with 3 targets of varying difficulty.

    - target_1: CC (easy, one-step from C)
    - target_2: CCC (medium, one-step from CC or two-step from C)
    - target_3: CCCC (hard, requires multiple steps)
    """
    targets = {}

    # Target 1: Simple one-step synthesis
    gt_route_1 = _make_simple_route("CC", "C", rank=1)
    targets["target_1"] = BenchmarkTarget(
        id="target_1",
        smiles=SmilesStr("CC"),
        inchi_key=_synthetic_inchikey("CC"),
        acceptable_routes=[gt_route_1],
    )

    # Target 2: Two-step synthesis
    gt_route_2 = _make_two_step_route("CCC", "CC", "C", rank=1)
    targets["target_2"] = BenchmarkTarget(
        id="target_2",
        smiles=SmilesStr("CCC"),
        inchi_key=_synthetic_inchikey("CCC"),
        acceptable_routes=[gt_route_2],
    )

    # Target 3: No ground truth (pure prediction)
    targets["target_3"] = BenchmarkTarget(
        id="target_3",
        smiles=SmilesStr("CCCC"),
        inchi_key=_synthetic_inchikey("CCCC"),
        acceptable_routes=[],
    )

    return BenchmarkSet(
        name="test-benchmark",
        description="Synthetic benchmark for testing",
        stock_name="test-stock",
        targets=targets,
    )


@pytest.fixture
def synthetic_predictions(synthetic_benchmark: BenchmarkSet) -> dict[str, list[Route]]:
    """
    Create model predictions for the synthetic benchmark.

    - target_1: 2 routes (one matching GT, one different)
    - target_2: 1 route (matching GT)
    - target_3: 1 route (no GT to match)
    """
    predictions = {}

    # Target 1: Two routes
    route_1a = _make_simple_route("CC", "C", rank=1)  # Matches GT
    route_1b = _make_simple_route("CC", "O", rank=2)  # Different leaf
    predictions["target_1"] = [route_1a, route_1b]

    # Target 2: One route matching GT
    route_2a = _make_two_step_route("CCC", "CC", "C", rank=1)
    predictions["target_2"] = [route_2a]

    # Target 3: One route
    route_3a = _make_two_step_route("CCCC", "CCC", "CC", rank=1)
    predictions["target_3"] = [route_3a]

    return predictions


@pytest.fixture
def minimal_stock() -> set[InchiKeyStr]:
    """Stock containing only C (methane)."""
    return {InchiKeyStr(_synthetic_inchikey("C"))}


@pytest.fixture
def extended_stock() -> set[InchiKeyStr]:
    """Stock containing C, CC, and O."""
    return {
        InchiKeyStr(_synthetic_inchikey("C")),
        InchiKeyStr(_synthetic_inchikey("CC")),
        InchiKeyStr(_synthetic_inchikey("O")),
    }


# =============================================================================
# Integration Tests: Ingest Workflow
# =============================================================================


class TestIngestModelPredictions:
    """Tests for the ingest_model_predictions workflow."""

    @pytest.mark.integration
    def test_ingest_basic_flow(self, tmp_path, synthetic_benchmark, synthetic_predictions):
        """Test basic ingestion saves routes to correct location."""
        adapter = SyntheticAdapter()

        # Convert predictions to serializable format (mode="json" handles set fields)
        raw_data = {
            target_id: [r.model_dump(mode="json") for r in routes]
            for target_id, routes in synthetic_predictions.items()
        }

        processed, save_path, stats = ingest_model_predictions(
            model_name="test-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
        )

        # Check stats
        assert stats.total_routes_in_raw_files == 3
        assert stats.num_targets_with_routes == 3
        assert stats.final_unique_routes_saved == 4  # 2 + 1 + 1

        # Check file was created
        assert save_path.exists()
        assert save_path.name == "routes.json.gz"

        # Verify saved data can be loaded
        loaded = load_routes(save_path)
        assert len(loaded) == 3
        assert len(loaded["target_1"]) == 2
        assert len(loaded["target_2"]) == 1
        assert len(loaded["target_3"]) == 1

    @pytest.mark.integration
    def test_ingest_with_missing_targets(self, tmp_path, synthetic_benchmark):
        """Test ingestion handles missing targets gracefully."""
        adapter = SyntheticAdapter()

        # Only provide predictions for target_1
        raw_data = {
            "target_1": [_make_simple_route("CC", "C", rank=1).model_dump(mode="json")],
        }

        processed, save_path, stats = ingest_model_predictions(
            model_name="partial-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
        )

        assert stats.total_routes_in_raw_files == 1
        assert stats.num_targets_with_routes == 1

        # Missing targets should have empty lists
        assert processed["target_2"] == []
        assert processed["target_3"] == []

    @pytest.mark.integration
    def test_ingest_with_smiles_key_fallback(self, tmp_path, synthetic_benchmark):
        """Test ingestion can resolve by SMILES when ID not found."""
        adapter = SyntheticAdapter()

        # Key by SMILES instead of target ID
        raw_data = {
            "CC": [_make_simple_route("CC", "C", rank=1).model_dump(mode="json")],
            "CCC": [_make_two_step_route("CCC", "CC", "C", rank=1).model_dump(mode="json")],
        }

        processed, save_path, stats = ingest_model_predictions(
            model_name="smiles-keyed-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
        )

        assert stats.total_routes_in_raw_files == 2
        assert len(processed["target_1"]) == 1
        assert len(processed["target_2"]) == 1
        assert processed["target_3"] == []

    @pytest.mark.integration
    def test_ingest_with_anonymization(self, tmp_path, synthetic_benchmark, synthetic_predictions):
        """Test ingestion with anonymized model names."""
        adapter = SyntheticAdapter()

        raw_data = {
            target_id: [r.model_dump(mode="json") for r in routes]
            for target_id, routes in synthetic_predictions.items()
        }

        _, save_path, _ = ingest_model_predictions(
            model_name="secret-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
            anonymize=True,
        )

        # Path should contain hash, not model name
        assert "secret-model" not in str(save_path)
        assert save_path.exists()

    @pytest.mark.integration
    def test_ingest_with_top_k_sampling(self, tmp_path, synthetic_benchmark):
        """Test ingestion with top-k sampling strategy."""
        adapter = SyntheticAdapter()

        # Create 5 routes for target_1 with different leaves (so they don't deduplicate)
        # Use different leaf SMILES to create unique signatures
        leaves = ["C", "O", "N", "S", "F"]
        routes = [_make_simple_route("CC", leaf, rank=i) for i, leaf in enumerate(leaves, start=1)]
        raw_data = {
            "target_1": [r.model_dump(mode="json") for r in routes],
        }

        processed, _, stats = ingest_model_predictions(
            model_name="sampled-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
            sampling_strategy="top-k",
            sample_k=3,
        )

        # Should only keep top 3
        assert len(processed["target_1"]) == 3
        assert stats.final_unique_routes_saved == 3

    @pytest.mark.integration
    def test_ingest_invalid_sampling_strategy_raises(self, tmp_path, synthetic_benchmark):
        """Test that invalid sampling strategy raises ValueError."""
        adapter = SyntheticAdapter()

        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            ingest_model_predictions(
                model_name="test",
                benchmark=synthetic_benchmark,
                raw_data={},
                adapter=adapter,
                output_dir=tmp_path,
                sampling_strategy="invalid-strategy",
                sample_k=3,
            )

    @pytest.mark.integration
    def test_ingest_sampling_without_k_raises(self, tmp_path, synthetic_benchmark):
        """Test that sampling without sample_k raises ValueError."""
        adapter = SyntheticAdapter()

        with pytest.raises(ValueError, match="Must provide sample_k"):
            ingest_model_predictions(
                model_name="test",
                benchmark=synthetic_benchmark,
                raw_data={},
                adapter=adapter,
                output_dir=tmp_path,
                sampling_strategy="top-k",
                sample_k=None,
            )


# =============================================================================
# Integration Tests: Score Workflow
# =============================================================================


class TestScoreModel:
    """Tests for the score_model workflow."""

    @pytest.mark.integration
    def test_score_basic_flow(self, synthetic_benchmark, synthetic_predictions, minimal_stock):
        """Test basic scoring produces correct evaluation results."""
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=synthetic_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="test-model",
        )

        assert eval_results.model_name == "test-model"
        assert eval_results.benchmark_name == "test-benchmark"
        assert eval_results.stock_name == "minimal"
        assert len(eval_results.results) == 3

    @pytest.mark.integration
    def test_score_solvability_with_minimal_stock(self, synthetic_benchmark, synthetic_predictions, minimal_stock):
        """Test solvability scoring with minimal stock (only C)."""
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=synthetic_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="test-model",
        )

        # target_1: CC <- C (solvable, leaf is C)
        t1 = eval_results.results["target_1"]
        assert t1.is_solvable is True
        assert t1.routes[0].is_solved is True  # First route uses C
        assert t1.routes[1].is_solved is False  # Second route uses O

        # target_2: CCC <- CC <- C (solvable, leaf is C)
        t2 = eval_results.results["target_2"]
        assert t2.is_solvable is True

        # target_3: CCCC <- CCC <- CC (not solvable, leaf is CC not in stock)
        t3 = eval_results.results["target_3"]
        assert t3.is_solvable is False

    @pytest.mark.integration
    def test_score_solvability_with_extended_stock(self, synthetic_benchmark, synthetic_predictions, extended_stock):
        """Test solvability with extended stock makes more routes solvable."""
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=synthetic_predictions,
            stock=extended_stock,
            stock_name="extended",
            model_name="test-model",
        )

        # target_1: Both routes solvable (C and O in stock)
        t1 = eval_results.results["target_1"]
        assert all(r.is_solved for r in t1.routes)

        # target_3: Now solvable (CC is in stock)
        t3 = eval_results.results["target_3"]
        assert t3.is_solvable is True

    @pytest.mark.integration
    def test_score_gt_match_detection(self, synthetic_benchmark, synthetic_predictions, minimal_stock):
        """Test ground truth match detection."""
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=synthetic_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="test-model",
        )

        # target_1: First route matches acceptable
        t1 = eval_results.results["target_1"]
        assert t1.routes[0].matches_acceptable is True
        assert t1.routes[1].matches_acceptable is False
        assert t1.acceptable_rank == 1  # First solved route is acceptable match

        # target_2: Route matches acceptable
        t2 = eval_results.results["target_2"]
        assert t2.routes[0].matches_acceptable is True
        assert t2.acceptable_rank == 1

        # target_3: No acceptable routes defined
        t3 = eval_results.results["target_3"]
        assert t3.routes[0].matches_acceptable is False
        assert t3.acceptable_rank is None

    @pytest.mark.integration
    def test_score_empty_predictions(self, synthetic_benchmark, minimal_stock):
        """Test scoring handles empty predictions gracefully."""
        empty_predictions = {
            "target_1": [],
            "target_2": [],
            "target_3": [],
        }

        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=empty_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="empty-model",
        )

        for target_id in ["target_1", "target_2", "target_3"]:
            t = eval_results.results[target_id]
            assert t.is_solvable is False
            assert t.acceptable_rank is None
            assert len(t.routes) == 0

    @pytest.mark.integration
    def test_score_missing_predictions_use_empty(self, synthetic_benchmark, minimal_stock):
        """Test scoring uses empty list for missing target predictions."""
        partial_predictions = {
            "target_1": [_make_simple_route("CC", "C", rank=1)],
            # target_2 and target_3 missing
        }

        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=partial_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="partial-model",
        )

        assert eval_results.results["target_1"].is_solvable is True
        assert eval_results.results["target_2"].is_solvable is False
        assert eval_results.results["target_3"].is_solvable is False

    @pytest.mark.integration
    def test_score_preserves_metadata(self, synthetic_benchmark, synthetic_predictions, minimal_stock):
        """Test that route metadata (length, convergence) is preserved."""
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=synthetic_predictions,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="test-model",
        )

        # Check metadata from benchmark target is copied
        t1 = eval_results.results["target_1"]
        assert t1.stratification_length == 1
        assert t1.stratification_is_convergent is False

        t2 = eval_results.results["target_2"]
        assert t2.stratification_length == 2
        assert t2.stratification_is_convergent is False

        t3 = eval_results.results["target_3"]
        assert t3.stratification_length is None
        assert t3.stratification_is_convergent is None


# =============================================================================
# Integration Tests: Full Pipeline
# =============================================================================


class TestFullPipeline:
    """End-to-end tests for the complete workflow."""

    @pytest.mark.integration
    def test_ingest_then_score_roundtrip(self, tmp_path, synthetic_benchmark, synthetic_predictions, minimal_stock):
        """Test complete ingest -> save -> load -> score pipeline."""
        adapter = SyntheticAdapter()

        # Convert to raw format
        raw_data = {
            target_id: [r.model_dump(mode="json") for r in routes]
            for target_id, routes in synthetic_predictions.items()
        }

        # Step 1: Ingest
        _, save_path, ingest_stats = ingest_model_predictions(
            model_name="pipeline-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
        )

        # Step 2: Load saved routes
        loaded_routes = load_routes(save_path)

        # Step 3: Score
        eval_results = score_model(
            benchmark=synthetic_benchmark,
            predictions=loaded_routes,
            stock=minimal_stock,
            stock_name="minimal",
            model_name="pipeline-model",
        )

        # Verify pipeline produces expected results
        assert ingest_stats.final_unique_routes_saved == 4
        assert len(eval_results.results) == 3

        # Check specific results
        assert eval_results.results["target_1"].is_solvable is True
        assert eval_results.results["target_2"].is_solvable is True
        assert eval_results.results["target_3"].is_solvable is False

    @pytest.mark.integration
    def test_pipeline_with_deduplication(self, tmp_path, synthetic_benchmark):
        """Test that duplicate routes are deduplicated during ingestion."""
        adapter = SyntheticAdapter()

        # Create duplicate routes
        route = _make_simple_route("CC", "C", rank=1)
        raw_data = {
            "target_1": [
                route.model_dump(mode="json"),
                route.model_dump(mode="json"),  # Duplicate
                route.model_dump(mode="json"),  # Duplicate
            ],
        }

        processed, _, stats = ingest_model_predictions(
            model_name="dedup-model",
            benchmark=synthetic_benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=tmp_path,
        )

        # Should deduplicate to 1 route
        assert stats.successful_routes_before_dedup == 3
        assert stats.final_unique_routes_saved == 1
        assert len(processed["target_1"]) == 1

    @pytest.mark.integration
    def test_pipeline_multiple_models(self, tmp_path, synthetic_benchmark, minimal_stock):
        """Test running pipeline for multiple models creates separate outputs."""
        adapter = SyntheticAdapter()

        models = {
            "model-a": {"target_1": [_make_simple_route("CC", "C", rank=1).model_dump(mode="json")]},
            "model-b": {"target_1": [_make_simple_route("CC", "O", rank=1).model_dump(mode="json")]},
        }

        results = {}
        for model_name, raw_data in models.items():
            _, save_path, _ = ingest_model_predictions(
                model_name=model_name,
                benchmark=synthetic_benchmark,
                raw_data=raw_data,
                adapter=adapter,
                output_dir=tmp_path,
            )

            loaded = load_routes(save_path)
            eval_result = score_model(
                benchmark=synthetic_benchmark,
                predictions=loaded,
                stock=minimal_stock,
                stock_name="minimal",
                model_name=model_name,
            )
            results[model_name] = eval_result

        # model-a uses C (in stock), model-b uses O (not in stock)
        assert results["model-a"].results["target_1"].is_solvable is True
        assert results["model-b"].results["target_1"].is_solvable is False


# =============================================================================
# Unit Tests: Adapter
# =============================================================================


class TestSyntheticAdapter:
    """Unit tests for the test adapter."""

    @pytest.mark.unit
    def test_adapter_yields_routes(self):
        """Test adapter yields Route objects from dict list."""
        adapter = SyntheticAdapter()
        route = _make_simple_route("CC", "C", rank=1)
        raw_data = [route.model_dump(mode="json")]

        target = BenchmarkTarget(
            id="test",
            smiles=SmilesStr("CC"),
            inchi_key=_synthetic_inchikey("CC"),
            is_convergent=False,
            route_length=1,
        )

        results = list(adapter.cast(raw_data, target))
        assert len(results) == 1
        assert results[0].target.smiles == "CC"

    @pytest.mark.unit
    def test_adapter_handles_invalid_data(self):
        """Test adapter skips invalid data gracefully."""
        adapter = SyntheticAdapter()
        raw_data = [{"invalid": "data"}]

        target = BenchmarkTarget(
            id="test",
            smiles=SmilesStr("CC"),
            inchi_key=_synthetic_inchikey("CC"),
            is_convergent=False,
            route_length=1,
        )

        results = list(adapter.cast(raw_data, target))
        assert len(results) == 0

    @pytest.mark.unit
    def test_adapter_handles_non_list(self):
        """Test adapter handles non-list input."""
        adapter = SyntheticAdapter()

        target = BenchmarkTarget(
            id="test",
            smiles=SmilesStr("CC"),
            inchi_key=_synthetic_inchikey("CC"),
            is_convergent=False,
            route_length=1,
        )

        results = list(adapter.cast("not a list", target))
        assert len(results) == 0
