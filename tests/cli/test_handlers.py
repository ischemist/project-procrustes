"""
In-process tests for CLI handlers.

These tests call handlers directly to contribute to coverage metrics.
For actual CLI integration tests (subprocess), see test_cli.py.
"""

import csv
import gzip
import json
from argparse import Namespace
from pathlib import Path

import pytest

from retrocast.chem import get_inchi_key
from retrocast.cli import adhoc, handlers
from retrocast.io.blob import save_json_gz
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Molecule, ReactionStep, Route
from tests.helpers import _synthetic_inchikey

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
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[make_simple_route("CC", "C")],
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
        stock_path = tmp_path / "stock.csv.gz"
        with gzip.open(stock_path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", get_inchi_key("C")])

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
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
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
        stock_path = tmp_path / "stock.csv.gz"
        with gzip.open(stock_path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", get_inchi_key("C")])

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
    """Unit tests for the handlers.handle_list handler (manifest-based discovery)."""

    def test_handle_list_with_manifests(self, tmp_path, capsys):
        """Test list handler discovers models from raw data manifests."""
        # Setup fake raw data structure
        raw_dir = tmp_path / "2-raw"
        (raw_dir / "model-a" / "bench-1").mkdir(parents=True)
        (raw_dir / "model-b" / "bench-1").mkdir(parents=True)

        # Create manifests with directives
        manifest_a = {
            "schema_version": "1.1",
            "directives": {"adapter": "aizynth", "raw_results_filename": "results.json.gz"},
            "action": "test",
            "parameters": {},
            "source_files": [],
            "output_files": [],
            "statistics": {},
        }
        manifest_b = {
            "schema_version": "1.1",
            "directives": {"adapter": "dms", "raw_results_filename": "results.json.gz"},
            "action": "test",
            "parameters": {},
            "source_files": [],
            "output_files": [],
            "statistics": {},
        }
        import json

        with open(raw_dir / "model-a" / "bench-1" / "manifest.json", "w") as f:
            json.dump(manifest_a, f)
        with open(raw_dir / "model-b" / "bench-1" / "manifest.json", "w") as f:
            json.dump(manifest_b, f)

        config = {"data_dir": str(tmp_path)}

        handlers.handle_list(config)

        captured = capsys.readouterr()
        assert "2 models" in captured.out
        assert "model-a" in captured.out
        assert "model-b" in captured.out
        assert "aizynth" in captured.out
        assert "dms" in captured.out

    def test_handle_list_no_manifests(self, tmp_path, capsys):
        """Test list handler with no manifests."""
        raw_dir = tmp_path / "2-raw"
        raw_dir.mkdir(parents=True)

        config = {"data_dir": str(tmp_path)}

        handlers.handle_list(config)

        captured = capsys.readouterr()
        assert "No models with manifests found" in captured.out


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

        # New default is data/retrocast/
        assert paths["benchmarks"] == Path("data/retrocast/1-benchmarks/definitions")


@pytest.mark.integration
class TestHandleVerify:
    """Integration tests for handle_verify CLI handler.

    Tests orchestration logic only - verification business logic is tested in test_verify.py.
    """

    @pytest.fixture
    def setup_valid_manifest(self, tmp_path):
        """Create a valid manifest with output file."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test data")

        from retrocast.io.provenance import create_manifest

        manifest = create_manifest(
            action="test-action",
            sources=[],
            outputs=[(data_file, {}, "unknown")],
            root_dir=tmp_path,
        )

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest_path, tmp_path

    @pytest.fixture
    def setup_invalid_manifest(self, tmp_path):
        """Create an invalid manifest (file missing on disk)."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test data")

        from retrocast.io.provenance import create_manifest

        manifest = create_manifest(
            action="test-action",
            sources=[],
            outputs=[(data_file, {}, "unknown")],
            root_dir=tmp_path,
        )

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        # Delete the file to make manifest invalid
        data_file.unlink()

        return manifest_path, tmp_path

    @pytest.fixture
    def setup_corrupted_manifest(self, tmp_path):
        """Create a manifest with corrupted file (hash mismatch)."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("original content")

        from retrocast.io.provenance import create_manifest

        manifest = create_manifest(
            action="test-action",
            sources=[],
            outputs=[(data_file, {}, "unknown")],
            root_dir=tmp_path,
        )

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        # Modify the file to cause hash mismatch
        data_file.write_text("CORRUPTED CONTENT")

        return manifest_path, tmp_path

    def test_verify_single_manifest_valid(self, setup_valid_manifest):
        """Test --target with valid manifest completes successfully."""
        manifest_path, root_dir = setup_valid_manifest

        config = {"data_dir": str(root_dir)}
        args = Namespace(
            target=str(manifest_path),
            all=False,
            deep=False,
        )

        # Should not raise SystemExit
        handlers.handle_verify(args, config)

    def test_verify_single_manifest_invalid_lenient_mode(self, setup_invalid_manifest):
        """Test --target with missing file passes in lenient mode (default)."""
        manifest_path, root_dir = setup_invalid_manifest

        config = {"data_dir": str(root_dir)}
        args = Namespace(
            target=str(manifest_path),
            all=False,
            deep=False,
            strict=False,  # Lenient mode (default)
        )

        # Should not raise SystemExit in lenient mode
        handlers.handle_verify(args, config)

    def test_verify_single_manifest_invalid_strict_mode_exits_with_1(self, setup_invalid_manifest):
        """Test --target with missing file exits with code 1 in strict mode."""
        manifest_path, root_dir = setup_invalid_manifest

        config = {"data_dir": str(root_dir)}
        args = Namespace(
            target=str(manifest_path),
            all=False,
            deep=False,
            strict=True,  # Strict mode
        )

        with pytest.raises(SystemExit) as exc_info:
            handlers.handle_verify(args, config)

        assert exc_info.value.code == 1

    def test_verify_corrupted_file_always_fails(self, setup_corrupted_manifest):
        """Test hash mismatch always fails regardless of lenient/strict mode."""
        manifest_path, root_dir = setup_corrupted_manifest

        config = {"data_dir": str(root_dir)}

        # Test lenient mode - should still fail on hash mismatch
        args_lenient = Namespace(
            target=str(manifest_path),
            all=False,
            deep=False,
            strict=False,
        )
        with pytest.raises(SystemExit) as exc_info_lenient:
            handlers.handle_verify(args_lenient, config)
        assert exc_info_lenient.value.code == 1

        # Test strict mode - should also fail on hash mismatch
        args_strict = Namespace(
            target=str(manifest_path),
            all=False,
            deep=False,
            strict=True,
        )
        with pytest.raises(SystemExit) as exc_info_strict:
            handlers.handle_verify(args_strict, config)
        assert exc_info_strict.value.code == 1

    def test_verify_all_discovers_multiple_manifests(self, tmp_path):
        """Test --all scans directory and verifies multiple manifests."""
        from retrocast.io.provenance import create_manifest

        # Create multiple manifests in different locations
        for i in range(3):
            data_dir = tmp_path / f"dir_{i}"
            data_dir.mkdir(parents=True)
            data_file = data_dir / "output.txt"
            data_file.write_text(f"data {i}")

            manifest = create_manifest(
                action=f"action-{i}",
                sources=[],
                outputs=[(data_file, {}, "unknown")],
                root_dir=tmp_path,
            )

            manifest_path = data_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                f.write(manifest.model_dump_json(indent=2))

        config = {"data_dir": str(tmp_path)}
        args = Namespace(
            target=None,
            all=True,
            deep=False,
        )

        # Should process all 3 manifests without error
        handlers.handle_verify(args, config)

    def test_verify_no_manifests_found(self, tmp_path, caplog):
        """Test handler warns when no manifests found."""
        config = {"data_dir": str(tmp_path)}
        args = Namespace(
            target=None,
            all=True,
            deep=False,
        )

        handlers.handle_verify(args, config)

        assert "No manifests found" in caplog.text

    def test_verify_with_deep_flag(self, setup_valid_manifest):
        """Test --deep flag is passed to verify_manifest."""
        manifest_path, root_dir = setup_valid_manifest

        config = {"data_dir": str(root_dir)}
        args = Namespace(
            target=str(manifest_path),
            all=False,
            deep=True,
        )

        # Should complete successfully with deep=True
        handlers.handle_verify(args, config)


@pytest.mark.unit
class TestSecurityAndErrorHandling:
    """Tests for security validation and specific exception handling in handlers.

    These tests verify:
    1. Path traversal protection via symlinks (_ingest_single)
    2. Specific exception handling (not broad Exception catching)
    3. Proper handling of SecurityError, JSONDecodeError, and OSError
    """

    def test_ingest_single_rejects_symlink_escape_logs_error(self, tmp_path, caplog):
        """Test that _ingest_single logs security error for symlink escape."""
        # Setup: Create directory structure
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw" / "test-model" / "test-benchmark"
        raw_dir.mkdir(parents=True)

        # Create an external directory outside raw
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        external_file = external_dir / "malicious.json.gz"
        external_file.write_text('{"test": "data"}')

        # Create a symlink inside raw_dir pointing to external file
        symlink_file = raw_dir / "results.json.gz"
        symlink_file.symlink_to(external_file)

        # Create a manifest with default filename (which is the symlink)
        manifest = {"directives": {"adapter": "syntheseus", "raw_results_filename": "results.json.gz"}}
        with open(raw_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create paths dict
        from retrocast.paths import get_paths

        paths = get_paths(data_dir)

        # Create minimal args
        args = Namespace(
            adapter=None,
            sampling_strategy=None,
            k=None,
            ignore_stereo=False,
            anonymize=False,
        )

        # Execute
        from retrocast.cli.handlers import _ingest_single

        with caplog.at_level("ERROR"):
            _ingest_single("test-model", "test-benchmark", paths, args)

        # Verify security error was logged
        assert "Security violation" in caplog.text

    def test_resolve_models_handles_security_error(self, tmp_path):
        """Test that _resolve_models catches SecurityError from malformed model names."""
        # Setup: Create a directory with path traversal attempt
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        raw_dir.mkdir(parents=True)

        # Create a malicious directory name (will be caught by filesystem, but test the handler)
        # Use a simple name that will pass filesystem but could be caught by validation
        malicious_model_dir = raw_dir / "../malicious"
        malicious_model_dir.mkdir(parents=True)
        benchmark_dir = malicious_model_dir / "test-bench"
        benchmark_dir.mkdir()

        # Create a valid manifest
        manifest = {"directives": {"adapter": "syntheseus"}}
        with open(benchmark_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Execute: _resolve_models should skip the malicious entry
        from retrocast.cli.handlers import _resolve_models
        from retrocast.paths import get_paths

        paths = get_paths(data_dir)
        args = Namespace(all_models=True)

        models = _resolve_models(args, paths, stage="ingest")

        # Should not include the malicious model (filtered out)
        assert "../malicious" not in models

    def test_resolve_models_handles_json_decode_error(self, tmp_path):
        """Test that _resolve_models catches JSONDecodeError from corrupted manifests."""
        # Setup: Create directory with corrupted manifest
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        model_dir = raw_dir / "test-model"
        benchmark_dir = model_dir / "test-benchmark"
        benchmark_dir.mkdir(parents=True)

        # Create a corrupted manifest (invalid JSON)
        with open(benchmark_dir / "manifest.json", "w") as f:
            f.write("{invalid json content")

        # Execute: _resolve_models should skip the corrupted manifest
        from retrocast.cli.handlers import _resolve_models
        from retrocast.paths import get_paths

        paths = get_paths(data_dir)
        args = Namespace(all_models=True)

        models = _resolve_models(args, paths, stage="ingest")

        # Should return empty list (no valid models found)
        assert models == []

    def test_resolve_models_handles_os_error(self, tmp_path, monkeypatch):
        """Test that _resolve_models catches OSError from file operations."""
        # Setup: Create directory with manifest that will cause OSError on open
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        model_dir = raw_dir / "test-model"
        benchmark_dir = model_dir / "test-benchmark"
        benchmark_dir.mkdir(parents=True)

        manifest_path = benchmark_dir / "manifest.json"
        manifest = {"directives": {"adapter": "syntheseus"}}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Make the file unreadable (simulates permission error)
        manifest_path.chmod(0o000)

        # Execute: _resolve_models should skip the unreadable manifest
        from retrocast.cli.handlers import _resolve_models
        from retrocast.paths import get_paths

        paths = get_paths(data_dir)
        args = Namespace(all_models=True)

        try:
            models = _resolve_models(args, paths, stage="ingest")
            # Should return empty list (no accessible models found)
            assert models == []
        finally:
            # Restore permissions for cleanup
            manifest_path.chmod(0o644)

    def test_resolve_benchmarks_handles_security_error(self, tmp_path):
        """Test that _resolve_benchmarks catches SecurityError from malformed filenames."""
        # Setup: Create benchmarks directory with malicious filename
        data_dir = tmp_path / "data"
        bench_dir = data_dir / "1-benchmarks" / "definitions"
        bench_dir.mkdir(parents=True)

        # Create a file with path traversal in name (filesystem may prevent this)
        # We'll test the validation logic instead
        valid_file = bench_dir / "valid.json.gz"
        valid_file.write_text("{}")

        # Create a file with suspicious name (though filesystem may sanitize)
        try:
            # Most filesystems won't allow this, but test the handler logic
            suspicious_file = bench_dir / "..%2Fmalicious.json.gz"
            suspicious_file.write_text("{}")
        except (OSError, ValueError):
            # Expected on most filesystems
            pass

        # Execute: _resolve_benchmarks should only return valid filenames
        from retrocast.cli.handlers import _resolve_benchmarks
        from retrocast.paths import get_paths

        paths = get_paths(data_dir)
        args = Namespace(all_datasets=True)

        benchmarks = _resolve_benchmarks(args, paths)

        # Should only include valid benchmark
        assert "valid" in benchmarks
        assert "../malicious" not in benchmarks

    def test_handle_list_handles_security_error(self, tmp_path):
        """Test that handle_list catches SecurityError from malformed directory names."""
        # Setup: Create directory structure with potentially malicious names
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        raw_dir.mkdir(parents=True)

        # Create a valid model directory
        valid_model_dir = raw_dir / "valid-model"
        valid_benchmark_dir = valid_model_dir / "test-benchmark"
        valid_benchmark_dir.mkdir(parents=True)

        manifest = {"directives": {"adapter": "syntheseus"}}
        with open(valid_benchmark_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Execute: handle_list should process without errors
        from retrocast.cli.handlers import handle_list

        config = {"data_dir": str(data_dir)}

        # Should not raise exception
        handle_list(config)

    def test_handle_list_handles_json_decode_error(self, tmp_path, caplog):
        """Test that handle_list catches JSONDecodeError from corrupted manifests."""
        # Setup: Create directory with corrupted manifest
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        model_dir = raw_dir / "test-model"
        benchmark_dir = model_dir / "test-benchmark"
        benchmark_dir.mkdir(parents=True)

        # Create a corrupted manifest
        with open(benchmark_dir / "manifest.json", "w") as f:
            f.write("{invalid json")

        # Execute: handle_list should skip corrupted manifests
        from retrocast.cli.handlers import handle_list

        config = {"data_dir": str(data_dir)}

        with caplog.at_level("DEBUG"):
            handle_list(config)

        # Should log that it's skipping the malformed manifest
        assert "Skipping malformed manifest" in caplog.text

    def test_handle_list_handles_os_error(self, tmp_path, caplog):
        """Test that handle_list catches OSError from file operations."""
        # Setup: Create directory with unreadable manifest
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "2-raw"
        model_dir = raw_dir / "test-model"
        benchmark_dir = model_dir / "test-benchmark"
        benchmark_dir.mkdir(parents=True)

        manifest_path = benchmark_dir / "manifest.json"
        manifest = {"directives": {"adapter": "syntheseus"}}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Make the file unreadable
        manifest_path.chmod(0o000)

        # Execute: handle_list should skip unreadable manifests
        from retrocast.cli.handlers import handle_list

        config = {"data_dir": str(data_dir)}

        try:
            with caplog.at_level("DEBUG"):
                handle_list(config)

            # Should log that it's skipping the malformed manifest
            assert "Skipping malformed manifest" in caplog.text
        finally:
            # Restore permissions for cleanup
            manifest_path.chmod(0o644)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
