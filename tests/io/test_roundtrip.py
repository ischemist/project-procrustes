"""
Tests for IO roundtrip serialization and provenance hashing.

Philosophy: Data persistence is not optional. Content hashes must be:
- Deterministic: Same input always produces same hash
- Order-invariant: Dict key order / set iteration order shouldn't matter
- Content-sensitive: Any change in data must change the hash
"""

import csv
import gzip
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrocast.exceptions import RetroCastIOError
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import (
    BenchmarkResultsLoader,
    load_benchmark,
    load_routes,
    load_stock_file,
    save_routes,
)
from retrocast.io.provenance import (
    _calculate_benchmark_content_hash,
    _calculate_predictions_content_hash,
    calculate_file_hash,
    create_manifest,
    generate_model_hash,
)
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.evaluation import EvaluationResults, ScoredRoute, TargetEvaluation
from retrocast.models.stats import MetricResult, ModelStatistics, ReliabilityFlag, StratifiedMetric
from tests.helpers import _synthetic_inchikey

# =============================================================================
# Tests for calculate_file_hash
# =============================================================================


@pytest.mark.unit
class TestFileHash:
    """Tests for file hash computation."""

    def test_same_content_same_hash(self, tmp_path):
        """Identical files should produce identical hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        content = b"Hello, World!"
        file1.write_bytes(content)
        file2.write_bytes(content)

        assert calculate_file_hash(file1) == calculate_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Different content should produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_bytes(b"Hello")
        file2.write_bytes(b"World")

        assert calculate_file_hash(file1) != calculate_file_hash(file2)

    def test_hash_changes_on_modification(self, tmp_path):
        """Modifying a file should change its hash."""
        test_file = tmp_path / "mutable.txt"
        test_file.write_bytes(b"Original")

        hash_before = calculate_file_hash(test_file)

        test_file.write_bytes(b"Modified")

        hash_after = calculate_file_hash(test_file)

        assert hash_before != hash_after

    def test_nonexistent_file_returns_error_marker(self, tmp_path):
        """Non-existent file should return error marker, not raise."""
        missing_file = tmp_path / "does_not_exist.txt"
        result = calculate_file_hash(missing_file)
        assert result == "error-hashing-file"

    def test_hash_is_64_hex_characters(self, tmp_path):
        """SHA256 hash should be 64 hex characters."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        result = calculate_file_hash(test_file)

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# =============================================================================
# Tests for generate_model_hash
# =============================================================================


@pytest.mark.unit
class TestGenerateModelHash:
    """Tests for model name hashing."""

    def test_deterministic(self):
        """Same model name should always produce same hash."""
        hash1 = generate_model_hash("my-model")
        hash2 = generate_model_hash("my-model")
        assert hash1 == hash2

    def test_different_names_different_hashes(self):
        """Different model names should produce different hashes."""
        hash1 = generate_model_hash("model-a")
        hash2 = generate_model_hash("model-b")
        assert hash1 != hash2

    def test_format_includes_prefix(self):
        """Hash should include the retrocasted-model prefix."""
        result = generate_model_hash("test-model")
        assert result.startswith("retrocasted-model-")

    def test_format_has_8_char_suffix(self):
        """Hash suffix should be 8 characters."""
        result = generate_model_hash("test-model")
        suffix = result.replace("retrocasted-model-", "")
        assert len(suffix) == 8


# =============================================================================
# Tests for benchmark content hashing
# =============================================================================


@pytest.mark.unit
class TestBenchmarkContentHash:
    """Tests for BenchmarkSet content hashing."""

    def test_deterministic(self, synthetic_route_factory):
        """Same benchmark should always produce same hash."""
        route = synthetic_route_factory("linear", depth=1)
        target = BenchmarkTarget(
            id="test-001",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[route],
        )
        benchmark = BenchmarkSet(
            name="test",
            description="test",
            targets={"test-001": target},
        )

        hash1 = _calculate_benchmark_content_hash(benchmark)
        hash2 = _calculate_benchmark_content_hash(benchmark)

        assert hash1 == hash2

    def test_order_invariant_target_dict(self, synthetic_route_factory):
        """Hash should be independent of target dict insertion order."""
        route = synthetic_route_factory("linear", depth=1)

        # Create targets in one order
        targets1 = {}
        for i in range(5):
            smiles = f"{'C' * (i + 1)}"
            targets1[f"t{i}"] = BenchmarkTarget(
                id=f"t{i}",
                smiles=smiles,
                inchi_key=_synthetic_inchikey(smiles),
                acceptable_routes=[route] if i == 0 else [],
            )

        # Create targets in reverse order
        targets2 = {}
        for i in reversed(range(5)):
            smiles = f"{'C' * (i + 1)}"
            targets2[f"t{i}"] = BenchmarkTarget(
                id=f"t{i}",
                smiles=smiles,
                inchi_key=_synthetic_inchikey(smiles),
                acceptable_routes=[route] if i == 0 else [],
            )

        bench1 = BenchmarkSet(name="test", targets=targets1)
        bench2 = BenchmarkSet(name="test", targets=targets2)

        assert _calculate_benchmark_content_hash(bench1) == _calculate_benchmark_content_hash(bench2)

    def test_content_sensitive_smiles_change(self):
        """Changing a SMILES should change the hash."""
        target1 = BenchmarkTarget(id="t1", smiles="CC", inchi_key=_synthetic_inchikey("CC"), acceptable_routes=[])
        target2 = BenchmarkTarget(id="t1", smiles="CCC", inchi_key=_synthetic_inchikey("CCC"), acceptable_routes=[])

        bench1 = BenchmarkSet(name="test", targets={"t1": target1})
        bench2 = BenchmarkSet(name="test", targets={"t1": target2})

        assert _calculate_benchmark_content_hash(bench1) != _calculate_benchmark_content_hash(bench2)

    def test_content_sensitive_metadata_change(self):
        """Changing metadata should change the hash."""
        target1 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
            metadata={"source": "A"},
        )
        target2 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
            metadata={"source": "B"},
        )

        bench1 = BenchmarkSet(name="test", targets={"t1": target1})
        bench2 = BenchmarkSet(name="test", targets={"t1": target2})

        assert _calculate_benchmark_content_hash(bench1) != _calculate_benchmark_content_hash(bench2)

    def test_metadata_order_invariant(self):
        """Metadata dict key order shouldn't affect hash."""
        target1 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
            metadata={"a": 1, "b": 2, "c": 3},
        )
        target2 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[],
            metadata={"c": 3, "a": 1, "b": 2},
        )

        bench1 = BenchmarkSet(name="test", targets={"t1": target1})
        bench2 = BenchmarkSet(name="test", targets={"t1": target2})

        assert _calculate_benchmark_content_hash(bench1) == _calculate_benchmark_content_hash(bench2)


# =============================================================================
# Tests for predictions content hashing
# =============================================================================


@pytest.mark.unit
class TestPredictionsContentHash:
    """Tests for route predictions content hashing."""

    def test_deterministic(self, synthetic_route_factory):
        """Same predictions should always produce same hash."""
        route = synthetic_route_factory("linear", depth=2)
        routes = {"target_1": [route]}

        hash1 = _calculate_predictions_content_hash(routes)
        hash2 = _calculate_predictions_content_hash(routes)

        assert hash1 == hash2

    def test_order_invariant_target_keys(self, synthetic_route_factory):
        """Hash should be independent of target dict key order."""
        route = synthetic_route_factory("linear", depth=1)

        # Create in different orders
        routes1 = {"a": [route], "b": [route], "c": [route]}
        routes2 = {"c": [route], "a": [route], "b": [route]}

        assert _calculate_predictions_content_hash(routes1) == _calculate_predictions_content_hash(routes2)

    def test_content_sensitive_rank_change(self, synthetic_route_factory):
        """Changing route rank should change hash."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=1)
        route2.rank = 2

        routes1 = {"t1": [route1]}
        routes2 = {"t1": [route2]}

        assert _calculate_predictions_content_hash(routes1) != _calculate_predictions_content_hash(routes2)

    def test_content_sensitive_different_routes(self, synthetic_route_factory):
        """Different route topologies should produce different hashes."""
        linear = synthetic_route_factory("linear", depth=2)
        convergent = synthetic_route_factory("convergent", depth=2)

        routes1 = {"t1": [linear]}
        routes2 = {"t1": [convergent]}

        assert _calculate_predictions_content_hash(routes1) != _calculate_predictions_content_hash(routes2)


# =============================================================================
# Tests for create_manifest
# =============================================================================


@pytest.mark.integration
class TestCreateManifest:
    """Tests for manifest creation."""

    def test_creates_manifest_with_source_file(self, tmp_path):
        """Manifest should include source file info."""
        source = tmp_path / "source.txt"
        source.write_text("source content")

        output = tmp_path / "output.txt"
        output.write_text("output content")

        manifest = create_manifest(
            action="test",
            sources=[source],
            outputs=[(output, {"key": "value"}, "unknown")],
            root_dir=tmp_path,
        )

        assert len(manifest.source_files) == 1
        assert manifest.source_files[0].path == "source.txt"
        assert manifest.source_files[0].file_hash != "error-hashing-file"

    def test_creates_manifest_with_benchmark_content_hash(self, tmp_path, synthetic_route_factory):
        """Manifest should include content hash for BenchmarkSet outputs."""
        route = synthetic_route_factory("linear", depth=1)
        target = BenchmarkTarget(
            id="t1",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[route],
        )
        benchmark = BenchmarkSet(name="test", targets={"t1": target})

        output_path = tmp_path / "benchmark.json.gz"
        save_json_gz(benchmark, output_path)

        manifest = create_manifest(
            action="test",
            sources=[],
            outputs=[(output_path, benchmark, "benchmark")],
            root_dir=tmp_path,
        )

        assert len(manifest.output_files) == 1
        assert manifest.output_files[0].content_hash is not None
        assert len(manifest.output_files[0].content_hash) == 64

    def test_creates_manifest_with_routes_content_hash(self, tmp_path, synthetic_route_factory):
        """Manifest should include content hash for route dict outputs."""
        route = synthetic_route_factory("linear", depth=1)
        routes = {"target_1": [route]}

        output_path = tmp_path / "routes.json.gz"
        save_routes(routes, output_path)

        manifest = create_manifest(
            action="test",
            sources=[],
            outputs=[(output_path, routes, "predictions")],
            root_dir=tmp_path,
        )

        assert len(manifest.output_files) == 1
        assert manifest.output_files[0].content_hash is not None

    def test_manifest_includes_parameters(self, tmp_path):
        """Manifest should include parameters."""
        output = tmp_path / "out.txt"
        output.write_text("test")

        manifest = create_manifest(
            action="test",
            sources=[],
            outputs=[(output, {}, "unknown")],
            root_dir=tmp_path,
            parameters={"key": "value", "number": 42},
        )

        assert manifest.parameters == {"key": "value", "number": 42}

    def test_manifest_includes_statistics(self, tmp_path):
        """Manifest should include statistics."""
        output = tmp_path / "out.txt"
        output.write_text("test")

        manifest = create_manifest(
            action="test",
            sources=[],
            outputs=[(output, {}, "unknown")],
            root_dir=tmp_path,
            statistics={"count": 100, "rate": 0.95},
        )

        assert manifest.statistics == {"count": 100, "rate": 0.95}


# =============================================================================
# Tests for route save/load roundtrip
# =============================================================================


@pytest.mark.integration
class TestRouteRoundtrip:
    """Tests for Route dictionary serialization."""

    def test_single_route_roundtrip(self, tmp_path, synthetic_route_factory):
        """Single route should survive roundtrip."""
        route = synthetic_route_factory("linear", depth=2)
        routes = {"target_1": [route]}
        path = tmp_path / "routes.json.gz"

        save_routes(routes, path)
        loaded = load_routes(path)

        assert len(loaded) == 1
        assert "target_1" in loaded
        assert len(loaded["target_1"]) == 1

        loaded_route = loaded["target_1"][0]
        assert loaded_route.target.smiles == route.target.smiles
        assert loaded_route.rank == route.rank
        assert loaded_route.length == route.length

    def test_empty_routes_dict(self, tmp_path):
        """Empty routes dictionary should roundtrip."""
        routes = {}
        path = tmp_path / "empty_routes.json.gz"

        save_routes(routes, path)
        loaded = load_routes(path)

        assert loaded == {}

    def test_multiple_routes_preserve_rank_order(self, tmp_path, synthetic_route_factory):
        """Multiple routes for same target should preserve ranks."""
        route1 = synthetic_route_factory("linear", depth=1)
        route2 = synthetic_route_factory("linear", depth=2)
        route3 = synthetic_route_factory("linear", depth=3)
        route1.rank = 1
        route2.rank = 2
        route3.rank = 3

        routes = {"target": [route1, route2, route3]}
        path = tmp_path / "routes.json.gz"

        save_routes(routes, path)
        loaded = load_routes(path)

        assert [r.rank for r in loaded["target"]] == [1, 2, 3]


# =============================================================================
# Tests for benchmark save/load roundtrip
# =============================================================================


@pytest.mark.integration
class TestBenchmarkRoundtrip:
    """Tests for BenchmarkSet serialization."""

    def test_benchmark_roundtrip(self, tmp_path, synthetic_route_factory):
        """BenchmarkSet should survive roundtrip with all fields."""
        route = synthetic_route_factory("linear", depth=1)
        target = BenchmarkTarget(
            id="test-001",
            smiles="CC",
            inchi_key=_synthetic_inchikey("CC"),
            acceptable_routes=[route],
            metadata={"source": "test"},
        )
        benchmark = BenchmarkSet(
            name="test-benchmark",
            description="Test benchmark",
            stock_name="test-stock",
            targets={"test-001": target},
        )
        path = tmp_path / "benchmark.json.gz"

        save_json_gz(benchmark, path)
        loaded = load_benchmark(path)

        assert loaded.name == benchmark.name
        assert loaded.description == benchmark.description
        assert loaded.stock_name == benchmark.stock_name
        assert len(loaded.targets) == 1

        loaded_target = loaded.targets["test-001"]
        assert loaded_target.smiles == "CC"
        assert loaded_target.metadata == {"source": "test"}
        assert loaded_target.primary_route is not None
        assert loaded_target.primary_route.length == 1


# =============================================================================
# Tests for stock file loading
# =============================================================================


@pytest.mark.unit
class TestStockFile:
    """Tests for stock file operations."""

    def test_load_stock_csv_returns_inchi_keys(self, tmp_path):
        """Stock CSV.GZ file should load as set of InChI keys."""
        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])
            writer.writerow(["CC", "OTMSDBZUPAUEDD-UHFFFAOYSA-N"])
            writer.writerow(["CCC", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"])

        stock = load_stock_file(stock_file)

        assert stock == {
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
            "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
        }

    def test_load_stock_csv_strips_whitespace(self, tmp_path):
        """Whitespace should be stripped from CSV.GZ entries."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "  VNWKTOKETHGBQD-UHFFFAOYSA-N  "])
            writer.writerow(["CC", " OTMSDBZUPAUEDD-UHFFFAOYSA-N\t"])

        stock = load_stock_file(stock_file)

        assert stock == {
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
        }

    def test_load_stock_csv_ignores_empty_inchikeys(self, tmp_path):
        """Empty InChI keys should be ignored."""
        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])
            writer.writerow(["", ""])  # Empty row
            writer.writerow(["CC", "OTMSDBZUPAUEDD-UHFFFAOYSA-N"])

        stock = load_stock_file(stock_file)

        assert stock == {
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
        }

    def test_load_stock_invalid_extension_raises(self, tmp_path):
        """Non-CSV.GZ files should raise RetroCastIOError."""

        stock_file = tmp_path / "stock.txt"
        stock_file.write_text("C\nCC\n")

        with pytest.raises(RetroCastIOError, match="Only .csv.gz format is supported"):
            load_stock_file(stock_file)

    def test_load_stock_csv_invalid_header_raises(self, tmp_path):
        """Invalid CSV.GZ header should raise RetroCastIOError."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt") as f:
            f.write("WrongHeader1,WrongHeader2\nC,VNWKTOKETHGBQD-UHFFFAOYSA-N\n")

        with pytest.raises(RetroCastIOError, match="Invalid stock CSV format"):
            load_stock_file(stock_file)

    def test_load_stock_missing_file_raises(self, tmp_path):
        """Missing file should raise RetroCastIOError."""

        stock_file = tmp_path / "nonexistent.csv.gz"

        with pytest.raises(RetroCastIOError, match="Stock file not found"):
            load_stock_file(stock_file)

    def test_load_stock_as_smiles_returns_smiles(self, tmp_path):
        """Stock CSV.GZ with return_as='smiles' should return SMILES set."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])
            writer.writerow(["CC", "OTMSDBZUPAUEDD-UHFFFAOYSA-N"])
            writer.writerow(["CCC", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"])

        stock = load_stock_file(stock_file, return_as="smiles")

        assert stock == {"C", "CC", "CCC"}

    def test_load_stock_as_smiles_strips_whitespace(self, tmp_path):
        """SMILES loading should strip whitespace."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["  C  ", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])
            writer.writerow([" CC\t", "OTMSDBZUPAUEDD-UHFFFAOYSA-N"])

        stock = load_stock_file(stock_file, return_as="smiles")

        assert stock == {"C", "CC"}

    def test_load_stock_as_smiles_ignores_empty_entries(self, tmp_path):
        """Empty SMILES should be ignored."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])
            writer.writerow(["", ""])
            writer.writerow(["CC", "OTMSDBZUPAUEDD-UHFFFAOYSA-N"])

        stock = load_stock_file(stock_file, return_as="smiles")

        assert stock == {"C", "CC"}

    def test_load_stock_invalid_return_as_raises(self, tmp_path):
        """Invalid return_as parameter should raise RetroCastIOError."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])
            writer.writerow(["C", "VNWKTOKETHGBQD-UHFFFAOYSA-N"])

        with pytest.raises(RetroCastIOError, match="Invalid return_as parameter"):
            load_stock_file(stock_file, return_as="invalid")

    def test_load_stock_as_smiles_missing_smiles_column_raises(self, tmp_path):
        """Missing SMILES column should raise when return_as='smiles'."""

        stock_file = tmp_path / "stock.csv.gz"
        with gzip.open(stock_file, "wt") as f:
            f.write("InChIKey\nVNWKTOKETHGBQD-UHFFFAOYSA-N\n")

        with pytest.raises(RetroCastIOError, match="Expected header with 'SMILES' column"):
            load_stock_file(stock_file, return_as="smiles")


# =============================================================================
# Hypothesis tests for hash properties
# =============================================================================


@pytest.mark.unit
@given(
    model_names=st.lists(
        st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        min_size=2,
        max_size=10,
        unique=True,
    )
)
@settings(max_examples=50)
def test_generate_model_hash_collision_resistant(model_names):
    """Property: Different model names should produce different hashes."""
    hashes = [generate_model_hash(name) for name in model_names]
    # All hashes should be unique
    assert len(set(hashes)) == len(model_names)


@pytest.mark.unit
@given(
    data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        values=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
            st.booleans(),
        ),
        max_size=20,
    )
)
@settings(max_examples=50)
def test_json_gz_roundtrip_arbitrary_dict(data):
    """Property: Any JSON-serializable dict should roundtrip without loss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test.json.gz"
        save_json_gz(data, path)
        loaded = load_json_gz(path)
        assert loaded == data


# =============================================================================
# Tests for BenchmarkResultsLoader
# =============================================================================


@pytest.mark.integration
class TestBenchmarkResultsLoader:
    """Tests for BenchmarkResultsLoader directory-based loading."""

    def _create_mock_statistics(self, model_name: str, benchmark: str, stock: str) -> ModelStatistics:
        """Helper to create a minimal ModelStatistics object."""
        reliability = ReliabilityFlag(code="OK", message="Sufficient samples")
        metric_result = MetricResult(
            value=0.75,
            ci_lower=0.70,
            ci_upper=0.80,
            n_samples=100,
            reliability=reliability,
        )
        stratified_metric = StratifiedMetric(
            metric_name="solvability",
            overall=metric_result,
            by_group={},
        )
        return ModelStatistics(
            model_name=model_name,
            benchmark=benchmark,
            stock=stock,
            solvability=stratified_metric,
            top_k_accuracy={},
        )

    def _create_mock_evaluation(self, model_name: str, benchmark: str, stock: str) -> EvaluationResults:
        """Helper to create a minimal EvaluationResults object."""
        scored_route = ScoredRoute(rank=1, is_solved=True, matches_acceptable=True)
        target_eval = TargetEvaluation(
            target_id="test-001",
            routes=[scored_route],
            is_solvable=True,
            acceptable_rank=1,
            stratification_length=3,
            stratification_is_convergent=False,
        )
        return EvaluationResults(
            model_name=model_name,
            benchmark_name=benchmark,
            stock_name=stock,
            has_acceptable_routes=True,
            results={"test-001": target_eval},
        )

    def test_load_statistics_single_model(self, tmp_path):
        """Should load statistics for a single model from expected directory."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create directory structure: data/5-results/{benchmark}/{model}/{stock}/
        stats_path = tmp_path / "5-results" / "pharma" / "model-a" / "n5-stock"
        stats_path.mkdir(parents=True)

        # Save mock statistics
        stats = self._create_mock_statistics("model-a", "pharma", "n5-stock")
        save_json_gz(stats, stats_path / "statistics.json.gz")

        # Load and verify
        loaded = loader.load_statistics("pharma", ["model-a"], "n5-stock")

        assert len(loaded) == 1
        assert loaded[0].model_name == "model-a"
        assert loaded[0].benchmark == "pharma"
        assert loaded[0].solvability.overall.value == 0.75

    def test_load_statistics_multiple_models(self, tmp_path):
        """Should load statistics for multiple models."""
        loader = BenchmarkResultsLoader(tmp_path)

        models = ["model-a", "model-b", "model-c"]
        for model in models:
            stats_path = tmp_path / "5-results" / "pharma" / model / "n5-stock"
            stats_path.mkdir(parents=True)
            stats = self._create_mock_statistics(model, "pharma", "n5-stock")
            save_json_gz(stats, stats_path / "statistics.json.gz")

        loaded = loader.load_statistics("pharma", models, "n5-stock")

        assert len(loaded) == 3
        loaded_names = {s.model_name for s in loaded}
        assert loaded_names == set(models)

    def test_load_statistics_missing_file_logs_warning(self, tmp_path, caplog):
        """Should log warning and skip missing files."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create only one model
        stats_path = tmp_path / "5-results" / "pharma" / "model-a" / "n5-stock"
        stats_path.mkdir(parents=True)
        stats = self._create_mock_statistics("model-a", "pharma", "n5-stock")
        save_json_gz(stats, stats_path / "statistics.json.gz")

        # Request two models, but only one exists
        loaded = loader.load_statistics("pharma", ["model-a", "model-missing"], "n5-stock")

        assert len(loaded) == 1
        assert loaded[0].model_name == "model-a"
        assert "Missing statistics" in caplog.text
        assert "model-missing" in caplog.text

    def test_load_statistics_corrupted_json_logs_error(self, tmp_path, caplog):
        """Should log error and skip corrupted files."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create corrupted file
        stats_path = tmp_path / "5-results" / "pharma" / "model-bad" / "n5-stock"
        stats_path.mkdir(parents=True)
        corrupted_file = stats_path / "statistics.json.gz"

        # Write invalid JSON
        with gzip.open(corrupted_file, "wt") as f:
            f.write("{ invalid json here }")

        # Create a valid file too
        valid_path = tmp_path / "5-results" / "pharma" / "model-good" / "n5-stock"
        valid_path.mkdir(parents=True)
        stats = self._create_mock_statistics("model-good", "pharma", "n5-stock")
        save_json_gz(stats, valid_path / "statistics.json.gz")

        loaded = loader.load_statistics("pharma", ["model-bad", "model-good"], "n5-stock")

        # Should only load the valid one
        assert len(loaded) == 1
        assert loaded[0].model_name == "model-good"
        assert "Failed to load" in caplog.text
        assert "model-bad" in caplog.text

    def test_load_statistics_empty_list(self, tmp_path):
        """Should return empty list when no models requested."""
        loader = BenchmarkResultsLoader(tmp_path)
        loaded = loader.load_statistics("pharma", [], "n5-stock")
        assert loaded == []

    def test_load_statistics_different_stock(self, tmp_path):
        """Should load from correct stock subdirectory."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create statistics for different stocks
        for stock in ["n5-stock", "n10-stock"]:
            stats_path = tmp_path / "5-results" / "pharma" / "model-a" / stock
            stats_path.mkdir(parents=True)
            stats = self._create_mock_statistics("model-a", "pharma", stock)
            save_json_gz(stats, stats_path / "statistics.json.gz")

        # Load n10-stock specifically
        loaded = loader.load_statistics("pharma", ["model-a"], "n10-stock")

        assert len(loaded) == 1
        assert loaded[0].stock == "n10-stock"

    def test_load_evaluation_success(self, tmp_path):
        """Should load evaluation results for a single model."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create evaluation file
        eval_path = tmp_path / "4-scored" / "pharma" / "model-a" / "n5-stock"
        eval_path.mkdir(parents=True)

        evaluation = self._create_mock_evaluation("model-a", "pharma", "n5-stock")
        save_json_gz(evaluation, eval_path / "evaluation.json.gz")

        loaded = loader.load_evaluation("pharma", "model-a", "n5-stock")

        assert loaded is not None
        assert loaded.model_name == "model-a"
        assert loaded.benchmark_name == "pharma"
        assert "test-001" in loaded.results

    def test_load_evaluation_missing_file_returns_none(self, tmp_path, caplog):
        """Should return None and log warning when file doesn't exist."""
        loader = BenchmarkResultsLoader(tmp_path)

        loaded = loader.load_evaluation("pharma", "missing-model", "n5-stock")

        assert loaded is None
        assert "Missing evaluation" in caplog.text
        assert "missing-model" in caplog.text

    def test_load_evaluation_corrupted_json_returns_none(self, tmp_path, caplog):
        """Should return None and log error when JSON is corrupted."""
        loader = BenchmarkResultsLoader(tmp_path)

        eval_path = tmp_path / "4-scored" / "pharma" / "model-bad" / "n5-stock"
        eval_path.mkdir(parents=True)

        # Write invalid JSON
        with gzip.open(eval_path / "evaluation.json.gz", "wt") as f:
            f.write("{ not valid json }")

        loaded = loader.load_evaluation("pharma", "model-bad", "n5-stock")

        assert loaded is None
        assert "Failed to load" in caplog.text
        assert "model-bad" in caplog.text

    def test_load_evaluation_different_benchmark(self, tmp_path):
        """Should load from correct benchmark subdirectory."""
        loader = BenchmarkResultsLoader(tmp_path)

        # Create evaluations for different benchmarks
        for benchmark in ["pharma", "paroutes"]:
            eval_path = tmp_path / "4-scored" / benchmark / "model-a" / "n5-stock"
            eval_path.mkdir(parents=True)
            evaluation = self._create_mock_evaluation("model-a", benchmark, "n5-stock")
            save_json_gz(evaluation, eval_path / "evaluation.json.gz")

        # Load paroutes specifically
        loaded = loader.load_evaluation("paroutes", "model-a", "n5-stock")

        assert loaded is not None
        assert loaded.benchmark_name == "paroutes"

    def test_loader_initialization(self, tmp_path):
        """Should correctly initialize directory paths."""
        loader = BenchmarkResultsLoader(tmp_path)

        assert loader.root == tmp_path
        assert loader.results_dir == tmp_path / "5-results"
        assert loader.scored_dir == tmp_path / "4-scored"

    def test_load_statistics_preserves_order(self, tmp_path):
        """Should return statistics in the same order as requested models."""
        loader = BenchmarkResultsLoader(tmp_path)

        models = ["model-c", "model-a", "model-b"]
        for model in models:
            stats_path = tmp_path / "5-results" / "pharma" / model / "n5-stock"
            stats_path.mkdir(parents=True)
            stats = self._create_mock_statistics(model, "pharma", "n5-stock")
            save_json_gz(stats, stats_path / "statistics.json.gz")

        loaded = loader.load_statistics("pharma", models, "n5-stock")
        loaded_names = [s.model_name for s in loaded]

        # Order should match request order
        assert loaded_names == models
