"""
In-process tests for CLI handlers.

These tests call handlers directly to contribute to coverage metrics.
For actual CLI integration tests (subprocess), see test_cli.py.
"""

from argparse import Namespace
from pathlib import Path

import pytest

from retrocast.chem import get_inchi_key
from retrocast.cli import adhoc, handlers
from retrocast.io.blob import save_json_gz
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Molecule, ReactionStep, Route

# --- Helpers ---


def make_leaf_molecule(smiles: str) -> Molecule:
    """Create a leaf molecule (no synthesis step)."""
    return Molecule(smiles=smiles, inchikey=get_inchi_key(smiles))


def make_simple_route(target_smiles: str, leaf_smiles: str, rank: int = 1) -> Route:
    """Create a one-step route: target <- leaf."""
    leaf = make_leaf_molecule(leaf_smiles)
    step = ReactionStep(reactants=[leaf])
    target = Molecule(
        smiles=target_smiles,
        inchikey=get_inchi_key(target_smiles),
        synthesis_step=step,
    )
    return Route(target=target, rank=rank)


# --- Test Classes ---


@pytest.mark.integration
class TestHandleScoreFile:
    """Integration tests for the adhoc.handle_score_file handler.

    Tests file I/O, scoring logic, and error handling with actual files.
    """

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create temporary test files for scoring."""
        # Create benchmark
        target = BenchmarkTarget(
            id="test-1",
            smiles="CC",
            is_convergent=False,
            route_length=1,
            ground_truth=make_simple_route("CC", "C"),
        )
        benchmark = BenchmarkSet(
            name="test-benchmark",
            description="Test benchmark",
            targets={"test-1": target},
        )
        benchmark_path = tmp_path / "benchmark.json.gz"
        save_json_gz(benchmark, benchmark_path)

        # Create predictions
        route = make_simple_route("CC", "C")
        predictions = {"test-1": [route.model_dump(mode="json")]}
        routes_path = tmp_path / "routes.json.gz"
        save_json_gz(predictions, routes_path)

        # Create stock file
        stock_path = tmp_path / "stock.txt"
        stock_path.write_text("C\n")

        # Output path
        output_path = tmp_path / "output.json.gz"

        return {
            "benchmark": benchmark_path,
            "routes": routes_path,
            "stock": stock_path,
            "output": output_path,
        }

    def test_handle_score_file_basic(self, test_files):
        """Test basic score-file handler execution."""
        args = Namespace(
            benchmark=str(test_files["benchmark"]),
            routes=str(test_files["routes"]),
            stock=str(test_files["stock"]),
            output=str(test_files["output"]),
            model_name="test-model",
        )

        adhoc.handle_score_file(args)

        assert test_files["output"].exists()

        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(test_files["output"])
        results = EvaluationResults.model_validate(data)
        assert results.model_name == "test-model"
        assert "test-1" in results.results
        assert results.results["test-1"].is_solvable is True

    def test_handle_score_file_solvable_with_stock(self, test_files):
        """Test that route is solvable when leaf is in stock."""
        args = Namespace(
            benchmark=str(test_files["benchmark"]),
            routes=str(test_files["routes"]),
            stock=str(test_files["stock"]),
            output=str(test_files["output"]),
            model_name="adhoc-model",
        )

        adhoc.handle_score_file(args)

        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(test_files["output"])
        results = EvaluationResults.model_validate(data)
        assert results.results["test-1"].is_solvable is True

    def test_handle_score_file_unsolvable(self, tmp_path):
        """Test scoring with route that uses non-stock molecule."""
        # Create benchmark
        target = BenchmarkTarget(
            id="test-1",
            smiles="CC",
            is_convergent=False,
            route_length=1,
        )
        benchmark = BenchmarkSet(
            name="test",
            targets={"test-1": target},
        )
        benchmark_path = tmp_path / "benchmark.json.gz"
        save_json_gz(benchmark, benchmark_path)

        # Create predictions with molecule not in stock
        route = make_simple_route("CC", "O")  # oxygen not in stock
        predictions = {"test-1": [route.model_dump(mode="json")]}
        routes_path = tmp_path / "routes.json.gz"
        save_json_gz(predictions, routes_path)

        # Stock only has carbon
        stock_path = tmp_path / "stock.txt"
        stock_path.write_text("C\n")

        output_path = tmp_path / "output.json.gz"

        args = Namespace(
            benchmark=str(benchmark_path),
            routes=str(routes_path),
            stock=str(stock_path),
            output=str(output_path),
            model_name="adhoc-model",
        )

        adhoc.handle_score_file(args)

        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(output_path)
        results = EvaluationResults.model_validate(data)
        assert results.results["test-1"].is_solvable is False

    def test_handle_score_file_missing_benchmark(self, test_files):
        """Test handler exits with missing benchmark."""
        args = Namespace(
            benchmark="/nonexistent/benchmark.json.gz",
            routes=str(test_files["routes"]),
            stock=str(test_files["stock"]),
            output=str(test_files["output"]),
            model_name="adhoc-model",
        )

        with pytest.raises(SystemExit) as exc_info:
            adhoc.handle_score_file(args)
        assert exc_info.value.code == 1

    def test_handle_score_file_missing_routes(self, test_files):
        """Test handler exits with missing routes."""
        args = Namespace(
            benchmark=str(test_files["benchmark"]),
            routes="/nonexistent/routes.json.gz",
            stock=str(test_files["stock"]),
            output=str(test_files["output"]),
            model_name="adhoc-model",
        )

        with pytest.raises(SystemExit) as exc_info:
            adhoc.handle_score_file(args)
        assert exc_info.value.code == 1

    def test_handle_score_file_missing_stock(self, test_files):
        """Test handler exits with missing stock."""
        args = Namespace(
            benchmark=str(test_files["benchmark"]),
            routes=str(test_files["routes"]),
            stock="/nonexistent/stock.txt",
            output=str(test_files["output"]),
            model_name="adhoc-model",
        )

        with pytest.raises(SystemExit) as exc_info:
            adhoc.handle_score_file(args)
        assert exc_info.value.code == 1


@pytest.mark.unit
class TestHandleList:
    """Unit tests for the handlers.handle_list handler."""

    def test_handle_list_basic(self, capsys):
        """Test list handler with models."""
        config = {
            "models": {
                "model-a": {"adapter": "aizynth"},
                "model-b": {"adapter": "dms"},
            }
        }

        handlers.handle_list(config)

        captured = capsys.readouterr()
        assert "2 models" in captured.out
        assert "model-a" in captured.out
        assert "model-b" in captured.out
        assert "aizynth" in captured.out
        assert "dms" in captured.out

    def test_handle_list_empty(self, capsys):
        """Test list handler with no models."""
        config = {"models": {}}

        handlers.handle_list(config)

        captured = capsys.readouterr()
        assert "0 models" in captured.out


@pytest.mark.unit
class TestHandleInfo:
    """Unit tests for the handlers.handle_info handler."""

    def test_handle_info_existing_model(self, capsys):
        """Test info handler with existing model."""
        config = {
            "models": {
                "test-model": {
                    "adapter": "aizynth",
                    "description": "A test model",
                    "sampling": {"strategy": "top_k", "k": 5},
                }
            }
        }

        handlers.handle_info(config, "test-model")

        captured = capsys.readouterr()
        assert "aizynth" in captured.out

    def test_handle_info_missing_model(self, caplog):
        """Test info handler with missing model logs error."""
        config = {"models": {}}

        handlers.handle_info(config, "nonexistent")

        # Should complete without exception (logs error)
        assert "not found" in caplog.text


@pytest.mark.unit
class TestResolveHelpers:
    """Unit tests for helper functions in handlers."""

    def test_get_paths(self):
        """Test _get_paths returns expected structure."""
        config = {"data_dir": "/tmp/test-data"}
        paths = handlers._get_paths(config)

        assert paths["benchmarks"] == Path("/tmp/test-data/1-benchmarks/definitions")
        assert paths["stocks"] == Path("/tmp/test-data/1-benchmarks/stocks")
        assert paths["raw"] == Path("/tmp/test-data/2-raw")
        assert paths["processed"] == Path("/tmp/test-data/3-processed")
        assert paths["scored"] == Path("/tmp/test-data/4-scored")
        assert paths["results"] == Path("/tmp/test-data/5-results")

    def test_get_paths_default(self):
        """Test _get_paths with default data_dir."""
        config = {}
        paths = handlers._get_paths(config)

        assert paths["benchmarks"] == Path("data/1-benchmarks/definitions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
