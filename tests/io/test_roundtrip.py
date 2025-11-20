"""
Tests for IO roundtrip serialization and provenance hashing.

Philosophy: Data persistence is not optional. Content hashes must be:
- Deterministic: Same input always produces same hash
- Order-invariant: Dict key order / set iteration order shouldn't matter
- Content-sensitive: Any change in data must change the hash
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import (
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

pytestmark = [pytest.mark.unit, pytest.mark.integration]


# =============================================================================
# Tests for calculate_file_hash
# =============================================================================


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


class TestBenchmarkContentHash:
    """Tests for BenchmarkSet content hashing."""

    def test_deterministic(self, synthetic_route_factory):
        """Same benchmark should always produce same hash."""
        route = synthetic_route_factory("linear", depth=1)
        target = BenchmarkTarget(
            id="test-001",
            smiles="CC",
            is_convergent=False,
            route_length=1,
            ground_truth=route,
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
            targets1[f"t{i}"] = BenchmarkTarget(
                id=f"t{i}",
                smiles=f"{'C' * (i + 1)}",
                is_convergent=False,
                route_length=i + 1,
                ground_truth=route if i == 0 else None,
            )

        # Create targets in reverse order
        targets2 = {}
        for i in reversed(range(5)):
            targets2[f"t{i}"] = BenchmarkTarget(
                id=f"t{i}",
                smiles=f"{'C' * (i + 1)}",
                is_convergent=False,
                route_length=i + 1,
                ground_truth=route if i == 0 else None,
            )

        bench1 = BenchmarkSet(name="test", targets=targets1)
        bench2 = BenchmarkSet(name="test", targets=targets2)

        assert _calculate_benchmark_content_hash(bench1) == _calculate_benchmark_content_hash(bench2)

    def test_content_sensitive_smiles_change(self):
        """Changing a SMILES should change the hash."""
        target1 = BenchmarkTarget(id="t1", smiles="CC", is_convergent=False, route_length=1)
        target2 = BenchmarkTarget(id="t1", smiles="CCC", is_convergent=False, route_length=1)

        bench1 = BenchmarkSet(name="test", targets={"t1": target1})
        bench2 = BenchmarkSet(name="test", targets={"t1": target2})

        assert _calculate_benchmark_content_hash(bench1) != _calculate_benchmark_content_hash(bench2)

    def test_content_sensitive_metadata_change(self):
        """Changing metadata should change the hash."""
        target1 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            is_convergent=False,
            route_length=1,
            metadata={"source": "A"},
        )
        target2 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            is_convergent=False,
            route_length=1,
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
            is_convergent=False,
            route_length=1,
            metadata={"a": 1, "b": 2, "c": 3},
        )
        target2 = BenchmarkTarget(
            id="t1",
            smiles="CC",
            is_convergent=False,
            route_length=1,
            metadata={"c": 3, "a": 1, "b": 2},
        )

        bench1 = BenchmarkSet(name="test", targets={"t1": target1})
        bench2 = BenchmarkSet(name="test", targets={"t1": target2})

        assert _calculate_benchmark_content_hash(bench1) == _calculate_benchmark_content_hash(bench2)


# =============================================================================
# Tests for predictions content hashing
# =============================================================================


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
            outputs=[(output, {"key": "value"})],
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
            is_convergent=False,
            route_length=1,
            ground_truth=route,
        )
        benchmark = BenchmarkSet(name="test", targets={"t1": target})

        output_path = tmp_path / "benchmark.json.gz"
        save_json_gz(benchmark, output_path)

        manifest = create_manifest(
            action="test",
            sources=[],
            outputs=[(output_path, benchmark)],
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
            outputs=[(output_path, routes)],
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
            outputs=[(output, {})],
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
            outputs=[(output, {})],
            statistics={"count": 100, "rate": 0.95},
        )

        assert manifest.statistics == {"count": 100, "rate": 0.95}


# =============================================================================
# Tests for route save/load roundtrip
# =============================================================================


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


class TestBenchmarkRoundtrip:
    """Tests for BenchmarkSet serialization."""

    def test_benchmark_roundtrip(self, tmp_path, synthetic_route_factory):
        """BenchmarkSet should survive roundtrip with all fields."""
        route = synthetic_route_factory("linear", depth=1)
        target = BenchmarkTarget(
            id="test-001",
            smiles="CC",
            is_convergent=False,
            route_length=1,
            ground_truth=route,
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
        assert loaded_target.ground_truth is not None
        assert loaded_target.ground_truth.length == 1


# =============================================================================
# Tests for stock file loading
# =============================================================================


class TestStockFile:
    """Tests for stock file operations."""

    def test_load_stock_file(self, tmp_path):
        """Stock file should load as set of SMILES."""
        stock_file = tmp_path / "stock.txt"
        stock_file.write_text("C\nCC\nCCC\nO\n")

        stock = load_stock_file(stock_file)

        assert stock == {"C", "CC", "CCC", "O"}

    def test_stock_file_strips_whitespace(self, tmp_path):
        """Whitespace should be stripped from stock entries."""
        stock_file = tmp_path / "stock.txt"
        stock_file.write_text("  C  \nCC\t\n  CCC  \n")

        stock = load_stock_file(stock_file)

        assert stock == {"C", "CC", "CCC"}

    def test_stock_file_ignores_empty_lines(self, tmp_path):
        """Empty lines should be ignored in stock files."""
        stock_file = tmp_path / "stock.txt"
        stock_file.write_text("C\n\nCC\n\n\nCCC\n")

        stock = load_stock_file(stock_file)

        assert stock == {"C", "CC", "CCC"}


# =============================================================================
# Hypothesis tests for hash properties
# =============================================================================


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
