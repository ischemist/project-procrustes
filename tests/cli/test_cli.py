"""
Integration tests for the retrocast CLI.

These tests verify CLI commands work correctly with real files
and synthetic data. They follow the testing philosophy:
- No mocking
- Real file I/O
- Synthetic but valid chemistry
"""

import subprocess
import sys

import pytest

from retrocast.chem import get_inchi_key
from retrocast.io.blob import save_json_gz
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Molecule, Route

# --- Helpers ---


def make_leaf_molecule(smiles: str) -> Molecule:
    """Create a leaf molecule (no synthesis step)."""
    return Molecule(smiles=smiles, inchikey=get_inchi_key(smiles))


def make_simple_route(target_smiles: str, leaf_smiles: str, rank: int = 1) -> Route:
    """Create a one-step route: target <- leaf."""
    leaf = make_leaf_molecule(leaf_smiles)
    from retrocast.models.chem import ReactionStep

    step = ReactionStep(reactants=[leaf])
    target = Molecule(
        smiles=target_smiles,
        inchikey=get_inchi_key(target_smiles),
        synthesis_step=step,
    )
    return Route(target=target, rank=rank)


# --- Test Classes ---


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "retrocast.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "retrocast" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_subcommand_help(self):
        """Test that subcommand --help works."""
        for cmd in ["list", "info", "ingest", "score", "score-file", "analyze"]:
            result = subprocess.run(
                [sys.executable, "-m", "retrocast.cli.main", cmd, "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Failed for {cmd}: {result.stderr}"
            assert "usage" in result.stdout.lower() or cmd in result.stdout.lower()

    def test_cli_missing_command(self):
        """Test that missing command fails gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "retrocast.cli.main"],
            capture_output=True,
            text=True,
        )
        # argparse returns 2 for missing required arguments
        assert result.returncode == 2


class TestScoreFileCommand:
    """Test the score-file ad-hoc scoring command."""

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create temporary test files for scoring."""
        # Create benchmark
        target = BenchmarkTarget(
            id="test-1",
            smiles="CC",  # ethane
            is_convergent=False,
            route_length=1,
            ground_truth=make_simple_route("CC", "C"),
        )
        benchmark = BenchmarkSet(
            name="test-benchmark",
            description="Test benchmark for CLI tests",
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

    def test_score_file_basic(self, test_files):
        """Test basic score-file execution."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "score-file",
                "--benchmark",
                str(test_files["benchmark"]),
                "--routes",
                str(test_files["routes"]),
                "--stock",
                str(test_files["stock"]),
                "--output",
                str(test_files["output"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert test_files["output"].exists()

        # Verify output can be loaded
        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(test_files["output"])
        results = EvaluationResults.model_validate(data)
        assert results.model_name == "adhoc-model"
        assert "test-1" in results.results
        assert results.results["test-1"].is_solvable is True

    def test_score_file_custom_model_name(self, test_files):
        """Test score-file with custom model name."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "score-file",
                "--benchmark",
                str(test_files["benchmark"]),
                "--routes",
                str(test_files["routes"]),
                "--stock",
                str(test_files["stock"]),
                "--output",
                str(test_files["output"]),
                "--model-name",
                "my-custom-model",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(test_files["output"])
        results = EvaluationResults.model_validate(data)
        assert results.model_name == "my-custom-model"

    def test_score_file_missing_benchmark(self, test_files):
        """Test score-file fails with missing benchmark."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "score-file",
                "--benchmark",
                "/nonexistent/benchmark.json.gz",
                "--routes",
                str(test_files["routes"]),
                "--stock",
                str(test_files["stock"]),
                "--output",
                str(test_files["output"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_score_file_unsolvable_route(self, tmp_path):
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

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "score-file",
                "--benchmark",
                str(benchmark_path),
                "--routes",
                str(routes_path),
                "--stock",
                str(stock_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        from retrocast.io.blob import load_json_gz
        from retrocast.models.evaluation import EvaluationResults

        data = load_json_gz(output_path)
        results = EvaluationResults.model_validate(data)
        # Route should not be solvable since O is not in stock
        assert results.results["test-1"].is_solvable is False


class TestListCommand:
    """Test the list command with a minimal config."""

    def test_list_with_config(self, tmp_path):
        """Test list command with a config file."""
        # Create minimal config
        config = {
            "data_dir": str(tmp_path / "data"),
            "models": {
                "test-model": {
                    "adapter": "aizynthfinder",
                    "description": "Test model",
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "--config",
                str(config_path),
                "list",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "test-model" in result.stdout
        assert "aizynthfinder" in result.stdout

    def test_list_missing_config(self, tmp_path):
        """Test list command with missing config falls back to dev config or fails."""
        # Run from a directory without retrocast-config.yaml to ensure no fallback
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "--config",
                str(tmp_path / "nonexistent.yaml"),
                "list",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,  # Run from temp dir where no fallback config exists
        )
        # Should exit with error when no fallback available
        assert result.returncode == 1


class TestInfoCommand:
    """Test the info command."""

    def test_info_with_model(self, tmp_path):
        """Test info command shows model details."""
        # Create config
        config = {
            "data_dir": str(tmp_path / "data"),
            "models": {
                "test-model": {
                    "adapter": "aizynthfinder",
                    "description": "A test model for testing",
                    "sampling": {"strategy": "top_k", "k": 5},
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "--config",
                str(config_path),
                "info",
                "--model",
                "test-model",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "aizynthfinder" in result.stdout

    def test_info_missing_model(self, tmp_path):
        """Test info command with non-existent model."""
        config = {
            "data_dir": str(tmp_path / "data"),
            "models": {},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrocast.cli.main",
                "--config",
                str(config_path),
                "info",
                "--model",
                "nonexistent",
            ],
            capture_output=True,
            text=True,
        )
        # Should complete but show error
        assert result.returncode == 0  # info logs error but doesn't exit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
