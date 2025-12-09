"""
Tests for the verify workflow - manifest and data integrity verification.

Philosophy:
- No mocking - use tmp_path for file I/O
- Synthetic data - create simple manifest chains
- Test both unit functions and integration workflows
"""

from pathlib import Path

import pytest

from retrocast.io.provenance import create_manifest
from retrocast.models.provenance import Manifest
from retrocast.workflow.verify import (
    _build_provenance_graph,
    _verify_logical_chain,
    _verify_physical_integrity,
    verify_manifest,
)

# =============================================================================
# Fixtures - Helper Functions
# =============================================================================


def create_simple_manifest(
    action: str, output_files: list[Path], source_files: list[Path] | None = None, root_dir: Path | None = None
) -> Manifest:
    """Helper to create a minimal manifest for testing."""
    sources = source_files or []
    outputs = [(f, {}, "unknown") for f in output_files]
    return create_manifest(action=action, sources=sources, outputs=outputs, root_dir=root_dir or Path.cwd())


def write_manifest_to_disk(manifest: Manifest, path: Path) -> None:
    """Helper to write a manifest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))


# =============================================================================
# Unit Tests: _build_provenance_graph
# =============================================================================


@pytest.mark.unit
class TestBuildProvenanceGraph:
    """Tests for _build_provenance_graph internal function."""

    def test_single_manifest_no_dependencies(self, tmp_path):
        """Single manifest with no dependencies should return graph with one entry."""
        # Create a simple data file
        data_file = tmp_path / "output.txt"
        data_file.write_text("test data")

        # Create manifest
        manifest = create_simple_manifest("test-action", [data_file], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Build graph
        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=Path("manifest.json"))
        graph = _build_provenance_graph(manifest_path, tmp_path, report)

        assert len(graph) == 1
        assert Path("manifest.json") in graph
        assert graph[Path("manifest.json")].action == "test-action"
        assert report.is_valid

    def test_two_level_chain(self, tmp_path):
        """Two-level dependency chain should discover both manifests."""
        # Create primary source (like benchmark)
        primary_dir = tmp_path / "1-benchmarks"
        primary_dir.mkdir(parents=True)
        primary_file = primary_dir / "benchmark.json"
        primary_file.write_text('{"name": "test"}')

        # Create intermediate artifact
        intermediate_dir = tmp_path / "3-processed" / "model-a"
        intermediate_dir.mkdir(parents=True)
        intermediate_file = intermediate_dir / "routes.json"
        intermediate_file.write_text('{"routes": []}')

        # Create parent manifest for intermediate
        parent_manifest = create_simple_manifest("process", [intermediate_file], [primary_file], root_dir=tmp_path)
        parent_manifest_path = intermediate_dir / "manifest.json"
        write_manifest_to_disk(parent_manifest, parent_manifest_path)

        # Create final artifact
        final_dir = tmp_path / "4-scored" / "model-a"
        final_dir.mkdir(parents=True)
        final_file = final_dir / "scores.json"
        final_file.write_text('{"scores": []}')

        # Create child manifest referencing intermediate
        child_manifest = create_simple_manifest("score", [final_file], [intermediate_file], root_dir=tmp_path)
        child_manifest_path = final_dir / "manifest.json"
        write_manifest_to_disk(child_manifest, child_manifest_path)

        # Build graph starting from child
        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=Path("4-scored/model-a/manifest.json"))
        graph = _build_provenance_graph(child_manifest_path, tmp_path, report)

        # Should discover both child and parent
        assert len(graph) == 2
        assert Path("4-scored/model-a/manifest.json") in graph
        assert Path("3-processed/model-a/manifest.json") in graph
        assert report.is_valid

    def test_missing_manifest_in_chain(self, tmp_path):
        """Missing manifest in dependency chain should report FAIL."""
        # Create a file that appears to be generated
        generated_dir = tmp_path / "3-processed" / "model-a"
        generated_dir.mkdir(parents=True)
        generated_file = generated_dir / "routes.json"
        generated_file.write_text('{"routes": []}')

        # Create manifest referencing the generated file (but no parent manifest exists)
        final_dir = tmp_path / "4-scored"
        final_dir.mkdir(parents=True)
        final_file = final_dir / "scores.json"
        final_file.write_text('{"scores": []}')

        manifest = create_simple_manifest("score", [final_file], [generated_file], root_dir=tmp_path)
        manifest_path = final_dir / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Build graph
        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=Path("4-scored/manifest.json"))
        graph = _build_provenance_graph(manifest_path, tmp_path, report)

        # Should still return current manifest but report should have FAIL
        assert len(graph) >= 1
        assert not report.is_valid
        assert any(entry.level == "FAIL" for entry in report.issues)

    def test_malformed_manifest_json(self, tmp_path):
        """Malformed manifest JSON should report FAIL."""
        # Create valid data file
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        # Create malformed manifest
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{ invalid json }")

        # Try to build graph
        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=Path("manifest.json"))
        graph = _build_provenance_graph(manifest_path, tmp_path, report)

        assert len(graph) == 0
        assert not report.is_valid
        assert any("Failed to load or parse" in entry.message for entry in report.issues)


# =============================================================================
# Unit Tests: _verify_logical_chain
# =============================================================================


@pytest.mark.unit
class TestVerifyLogicalChain:
    """Tests for _verify_logical_chain internal function."""

    def test_consistent_hash_between_parent_and_child(self, tmp_path):
        """Consistent hash between parent output and child source should PASS."""
        # Create data file
        data_file = tmp_path / "3-processed" / "routes.json"
        data_file.parent.mkdir(parents=True)
        data_file.write_text('{"routes": []}')

        # Create parent manifest
        parent_manifest = create_simple_manifest("process", [data_file], root_dir=tmp_path)
        parent_path = Path("3-processed/manifest.json")

        # Create child that references parent's output
        child_manifest = create_simple_manifest("score", [], [data_file], root_dir=tmp_path)
        child_path = Path("4-scored/manifest.json")

        # Build graph
        graph = {parent_path: parent_manifest, child_path: child_manifest}

        # Verify chain
        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=child_path)
        _verify_logical_chain(graph, report)

        assert report.is_valid
        assert any("Link to parent manifest" in entry.message and entry.level == "PASS" for entry in report.issues)

    def test_hash_mismatch_detection(self, tmp_path):
        """Hash mismatch between parent and child should FAIL."""
        data_file = tmp_path / "3-processed" / "routes.json"
        data_file.parent.mkdir(parents=True)
        data_file.write_text('{"routes": []}')

        # Create parent with correct hash
        parent_manifest = create_simple_manifest("process", [data_file], root_dir=tmp_path)
        parent_path = Path("3-processed/manifest.json")

        # Create child with WRONG hash
        child_manifest = create_simple_manifest("score", [], [data_file], root_dir=tmp_path)
        # Manually corrupt the hash in child's source
        child_manifest.source_files[0].file_hash = "wrong-hash-value"
        child_path = Path("4-scored/manifest.json")

        graph = {parent_path: parent_manifest, child_path: child_manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=child_path)
        _verify_logical_chain(graph, report)

        assert not report.is_valid
        assert any("Hash mismatch" in entry.message and entry.level == "FAIL" for entry in report.issues)

    def test_primary_artifacts_skip_verification(self, tmp_path):
        """Primary artifacts (1-benchmarks, 2-raw) should be marked as PASS without verification."""
        # Create benchmark file
        benchmark_file = tmp_path / "1-benchmarks" / "test.json"
        benchmark_file.parent.mkdir(parents=True)
        benchmark_file.write_text('{"benchmark": "data"}')

        # Create manifest that uses benchmark
        manifest = create_simple_manifest("process", [], [benchmark_file], root_dir=tmp_path)
        manifest_path = Path("3-processed/manifest.json")

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=manifest_path)
        _verify_logical_chain(graph, report)

        # Should pass with note that it's a primary artifact
        assert report.is_valid
        assert any("primary artifact" in entry.message.lower() for entry in report.issues)

    def test_missing_parent_manifest(self, tmp_path):
        """Missing parent manifest should report WARN."""
        # File that appears to be generated
        data_file = tmp_path / "3-processed" / "routes.json"
        data_file.parent.mkdir(parents=True)
        data_file.write_text('{"routes": []}')

        # Child references it but parent manifest not in graph
        child_manifest = create_simple_manifest("score", [], [data_file], root_dir=tmp_path)
        child_path = Path("4-scored/manifest.json")

        graph = {child_path: child_manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=child_path)
        _verify_logical_chain(graph, report)

        # Should have warnings
        assert any(entry.level == "WARN" for entry in report.issues)


# =============================================================================
# Unit Tests: _verify_physical_integrity
# =============================================================================


@pytest.mark.unit
class TestVerifyPhysicalIntegrity:
    """Tests for _verify_physical_integrity internal function."""

    def test_file_matches_manifest_hash(self, tmp_path):
        """File with matching hash should PASS."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test content")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = Path("manifest.json")

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report)

        assert report.is_valid
        assert any("hash matches" in entry.message.lower() and entry.level == "PASS" for entry in report.issues)

    def test_file_hash_mismatch_detection(self, tmp_path):
        """File with wrong hash should FAIL."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("original content")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = Path("manifest.json")

        # Modify file after creating manifest
        data_file.write_text("MODIFIED CONTENT")

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report)

        assert not report.is_valid
        assert any("HASH MISMATCH" in entry.message and entry.level == "FAIL" for entry in report.issues)

    def test_missing_file_detection_strict_mode(self, tmp_path):
        """Missing file should FAIL in strict mode (lenient=False)."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = Path("manifest.json")

        # Delete file after creating manifest
        data_file.unlink()

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report, lenient=False)

        assert not report.is_valid
        assert any("MISSING from disk" in entry.message and entry.level == "FAIL" for entry in report.issues)

    def test_missing_file_detection_lenient_mode(self, tmp_path):
        """Missing file should WARN in lenient mode (lenient=True, default)."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = Path("manifest.json")

        # Delete file after creating manifest
        data_file.unlink()

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report, lenient=True)

        # Should still be valid (only warnings, no failures)
        assert report.is_valid
        assert any("MISSING from disk" in entry.message and entry.level == "WARN" for entry in report.issues)

    def test_multiple_files_in_graph(self, tmp_path):
        """Multiple files across manifests should all be verified."""
        # Create multiple files
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        file2 = tmp_path / "file2.txt"
        file2.write_text("content 2")

        manifest1 = create_simple_manifest("action1", [file1], root_dir=tmp_path)
        manifest2 = create_simple_manifest("action2", [file2], root_dir=tmp_path)

        graph = {Path("manifest1.json"): manifest1, Path("manifest2.json"): manifest2}

        from retrocast.models.provenance import VerificationReport

        report = VerificationReport(manifest_path=Path("manifest1.json"))
        _verify_physical_integrity(graph, tmp_path, report)

        # Should check both files
        assert report.is_valid
        pass_count = sum(1 for e in report.issues if e.level == "PASS" and "hash matches" in e.message.lower())
        assert pass_count == 2

    def test_hash_mismatch_always_fails_regardless_of_lenient(self, tmp_path):
        """Hash mismatch should ALWAYS fail, even in lenient mode."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("original content")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = Path("manifest.json")

        # Modify file after creating manifest
        data_file.write_text("MODIFIED CONTENT")

        graph = {manifest_path: manifest}

        from retrocast.models.provenance import VerificationReport

        # Test with lenient=True - should still fail
        report_lenient = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report_lenient, lenient=True)
        assert not report_lenient.is_valid
        assert any("HASH MISMATCH" in entry.message and entry.level == "FAIL" for entry in report_lenient.issues)

        # Test with lenient=False - should also fail
        report_strict = VerificationReport(manifest_path=manifest_path)
        _verify_physical_integrity(graph, tmp_path, report_strict, lenient=False)
        assert not report_strict.is_valid
        assert any("HASH MISMATCH" in entry.message and entry.level == "FAIL" for entry in report_strict.issues)


# =============================================================================
# Integration Tests: verify_manifest - Shallow Mode
# =============================================================================


@pytest.mark.integration
class TestVerifyManifestShallow:
    """Integration tests for shallow verification (deep=False)."""

    def test_valid_single_manifest_all_files_present(self, tmp_path):
        """Valid manifest with all files present should pass shallow verification."""
        # Create output files
        file1 = tmp_path / "output1.txt"
        file1.write_text("data 1")
        file2 = tmp_path / "output2.txt"
        file2.write_text("data 2")

        # Create manifest
        manifest = create_simple_manifest("test-action", [file1, file2], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Run shallow verification
        report = verify_manifest(manifest_path, tmp_path, deep=False)

        assert report.is_valid
        assert report.manifest_path == Path("manifest.json")
        # Should have PASS entries for both files
        pass_count = sum(1 for e in report.issues if e.level == "PASS")
        assert pass_count >= 2

    def test_detect_hash_mismatch_in_output_files(self, tmp_path):
        """Shallow verification should detect hash mismatch in output files."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("original")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Modify file
        data_file.write_text("modified")

        report = verify_manifest(manifest_path, tmp_path, deep=False)

        assert not report.is_valid
        assert any("HASH MISMATCH" in e.message for e in report.issues)

    def test_detect_missing_output_files_strict(self, tmp_path):
        """Shallow verification should detect missing output files in strict mode."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Delete file
        data_file.unlink()

        report = verify_manifest(manifest_path, tmp_path, deep=False, lenient=False)

        assert not report.is_valid
        assert any("MISSING" in e.message and e.level == "FAIL" for e in report.issues)

    def test_detect_missing_output_files_lenient(self, tmp_path):
        """Shallow verification should warn about missing output files in lenient mode."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        # Delete file
        data_file.unlink()

        report = verify_manifest(manifest_path, tmp_path, deep=False, lenient=True)

        assert report.is_valid  # Still valid in lenient mode
        assert any("MISSING" in e.message and e.level == "WARN" for e in report.issues)

    def test_malformed_manifest_fails_gracefully(self, tmp_path):
        """Shallow verification should handle malformed manifest gracefully."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{ bad json")

        report = verify_manifest(manifest_path, tmp_path, deep=False)

        assert not report.is_valid
        assert any("Failed to load" in e.message for e in report.issues)


# =============================================================================
# Integration Tests: verify_manifest - Deep Mode
# =============================================================================


@pytest.mark.integration
class TestVerifyManifestDeep:
    """Integration tests for deep verification (deep=True)."""

    def test_simple_two_level_chain_all_valid(self, tmp_path):
        """Valid 2-level dependency chain should pass deep verification."""
        # Create primary data (benchmark)
        primary_dir = tmp_path / "1-benchmarks"
        primary_dir.mkdir(parents=True)
        primary_file = primary_dir / "benchmark.json"
        primary_file.write_text('{"name": "test"}')

        # Create intermediate artifact
        intermediate_dir = tmp_path / "3-processed" / "model-a"
        intermediate_dir.mkdir(parents=True)
        intermediate_file = intermediate_dir / "routes.json"
        intermediate_file.write_text('{"routes": []}')

        # Create parent manifest
        parent_manifest = create_simple_manifest("ingest", [intermediate_file], [primary_file], root_dir=tmp_path)
        parent_manifest_path = intermediate_dir / "manifest.json"
        write_manifest_to_disk(parent_manifest, parent_manifest_path)

        # Create final artifact
        final_dir = tmp_path / "4-scored" / "model-a"
        final_dir.mkdir(parents=True)
        final_file = final_dir / "scores.json"
        final_file.write_text('{"scores": []}')

        # Create child manifest
        child_manifest = create_simple_manifest("score", [final_file], [intermediate_file], root_dir=tmp_path)
        child_manifest_path = final_dir / "manifest.json"
        write_manifest_to_disk(child_manifest, child_manifest_path)

        # Run deep verification from child
        report = verify_manifest(child_manifest_path, tmp_path, deep=True)

        assert report.is_valid
        # Should have INFO about graph discovery
        assert any("Graph Discovery" in e.message for e in report.issues)
        # Should have PASS for provenance graph build
        assert any("provenance graph with" in e.message.lower() and e.level == "PASS" for e in report.issues)

    def test_three_level_chain_with_primary_artifact(self, tmp_path):
        """3-level chain ending in primary artifact should verify correctly."""
        # Primary: benchmark
        benchmark_dir = tmp_path / "1-benchmarks"
        benchmark_dir.mkdir(parents=True)
        benchmark_file = benchmark_dir / "pharma.json"
        benchmark_file.write_text('{"targets": []}')

        # Level 1: processed routes
        processed_dir = tmp_path / "3-processed" / "model-x"
        processed_dir.mkdir(parents=True)
        routes_file = processed_dir / "routes.json"
        routes_file.write_text('{"routes": []}')

        manifest1 = create_simple_manifest("ingest", [routes_file], [benchmark_file], root_dir=tmp_path)
        manifest1_path = processed_dir / "manifest.json"
        write_manifest_to_disk(manifest1, manifest1_path)

        # Level 2: scored
        scored_dir = tmp_path / "4-scored" / "model-x"
        scored_dir.mkdir(parents=True)
        scores_file = scored_dir / "evaluation.json"
        scores_file.write_text('{"scores": []}')

        manifest2 = create_simple_manifest("score", [scores_file], [routes_file], root_dir=tmp_path)
        manifest2_path = scored_dir / "manifest.json"
        write_manifest_to_disk(manifest2, manifest2_path)

        # Level 3: final analysis
        results_dir = tmp_path / "5-results" / "model-x"
        results_dir.mkdir(parents=True)
        results_file = results_dir / "statistics.json"
        results_file.write_text('{"statistics": []}')

        manifest3 = create_simple_manifest("analyze", [results_file], [scores_file], root_dir=tmp_path)
        manifest3_path = results_dir / "manifest.json"
        write_manifest_to_disk(manifest3, manifest3_path)

        # Verify from deepest level
        report = verify_manifest(manifest3_path, tmp_path, deep=True)

        assert report.is_valid
        # Should discover all 3 manifests
        assert any("provenance graph with 3 manifests" in e.message.lower() for e in report.issues)

    def test_detect_provenance_break_hash_mismatch(self, tmp_path):
        """Deep verification should detect hash mismatch in provenance chain."""
        # Create parent artifact
        parent_dir = tmp_path / "3-processed"
        parent_dir.mkdir(parents=True)
        parent_file = parent_dir / "data.json"
        parent_file.write_text('{"data": "original"}')

        parent_manifest = create_simple_manifest("process", [parent_file], root_dir=tmp_path)
        parent_manifest_path = parent_dir / "manifest.json"
        write_manifest_to_disk(parent_manifest, parent_manifest_path)

        # Create child
        child_dir = tmp_path / "4-scored"
        child_dir.mkdir(parents=True)
        child_file = child_dir / "scores.json"
        child_file.write_text('{"scores": []}')

        child_manifest = create_simple_manifest("score", [child_file], [parent_file], root_dir=tmp_path)
        child_manifest_path = child_dir / "manifest.json"
        write_manifest_to_disk(child_manifest, child_manifest_path)

        # TAMPER: modify parent's output file
        parent_file.write_text('{"data": "TAMPERED"}')

        # Deep verification should catch this
        report = verify_manifest(child_manifest_path, tmp_path, deep=True)

        assert not report.is_valid
        assert any("HASH MISMATCH" in e.message for e in report.issues)

    def test_detect_missing_intermediate_manifest(self, tmp_path):
        """Deep verification should detect missing intermediate manifest in chain."""
        # Create intermediate file (appears to be generated)
        intermediate_dir = tmp_path / "3-processed"
        intermediate_dir.mkdir(parents=True)
        intermediate_file = intermediate_dir / "data.json"
        intermediate_file.write_text('{"data": []}')
        # NOTE: No manifest created for intermediate

        # Create final level that references it
        final_dir = tmp_path / "4-scored"
        final_dir.mkdir(parents=True)
        final_file = final_dir / "scores.json"
        final_file.write_text('{"scores": []}')

        final_manifest = create_simple_manifest("score", [final_file], [intermediate_file], root_dir=tmp_path)
        final_manifest_path = final_dir / "manifest.json"
        write_manifest_to_disk(final_manifest, final_manifest_path)

        # Deep verify should fail due to missing parent manifest
        report = verify_manifest(final_manifest_path, tmp_path, deep=True)

        assert not report.is_valid
        # Should have FAIL for missing manifest
        assert any(e.level == "FAIL" for e in report.issues)

    def test_complex_graph_with_multiple_sources(self, tmp_path):
        """Deep verification with manifest having multiple sources."""
        # Create multiple sources
        benchmark_dir = tmp_path / "1-benchmarks"
        benchmark_dir.mkdir(parents=True)
        benchmark_file = benchmark_dir / "benchmark.json"
        benchmark_file.write_text('{"benchmark": []}')

        stock_dir = tmp_path / "2-raw"
        stock_dir.mkdir(parents=True)
        stock_file = stock_dir / "stock.txt"
        stock_file.write_text("C\nCC\n")

        routes_dir = tmp_path / "3-processed" / "model-y"
        routes_dir.mkdir(parents=True)
        routes_file = routes_dir / "routes.json"
        routes_file.write_text('{"routes": []}')

        routes_manifest = create_simple_manifest("ingest", [routes_file], [benchmark_file], root_dir=tmp_path)
        routes_manifest_path = routes_dir / "manifest.json"
        write_manifest_to_disk(routes_manifest, routes_manifest_path)

        # Create final that uses multiple sources (benchmark, stock, routes)
        final_dir = tmp_path / "4-scored" / "model-y"
        final_dir.mkdir(parents=True)
        final_file = final_dir / "evaluation.json"
        final_file.write_text('{"evaluation": []}')

        final_manifest = create_simple_manifest(
            "score", [final_file], [benchmark_file, stock_file, routes_file], root_dir=tmp_path
        )
        final_manifest_path = final_dir / "manifest.json"
        write_manifest_to_disk(final_manifest, final_manifest_path)

        # Verify
        report = verify_manifest(final_manifest_path, tmp_path, deep=True)

        assert report.is_valid
        # Should discover routes manifest and verify primary sources
        assert any("primary artifact" in e.message.lower() for e in report.issues)


# =============================================================================
# Integration Tests: Verification Report
# =============================================================================


@pytest.mark.integration
class TestVerificationReport:
    """Tests for VerificationReport structure and properties."""

    def test_report_contains_expected_entries(self, tmp_path):
        """Report should contain PASS/FAIL/INFO entries as appropriate."""
        data_file = tmp_path / "output.txt"
        data_file.write_text("test")

        manifest = create_simple_manifest("test", [data_file], root_dir=tmp_path)
        manifest_path = tmp_path / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        report = verify_manifest(manifest_path, tmp_path, deep=False)

        # Check entry structure
        assert len(report.issues) > 0
        for entry in report.issues:
            assert entry.level in ["PASS", "FAIL", "WARN", "INFO"]
            assert isinstance(entry.message, str)
            assert len(entry.message) > 0

    def test_is_valid_property_works_correctly(self, tmp_path):
        """is_valid should be True only when no FAIL entries exist."""
        # Valid case
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("test")
        manifest1 = create_simple_manifest("test", [valid_file], root_dir=tmp_path)
        path1 = tmp_path / "manifest1.json"
        write_manifest_to_disk(manifest1, path1)

        report1 = verify_manifest(path1, tmp_path, deep=False)
        assert report1.is_valid

        # Invalid case in lenient mode - missing file (should still be valid, just warnings)
        invalid_file = tmp_path / "missing.txt"
        invalid_file.write_text("test")
        manifest2 = create_simple_manifest("test", [invalid_file], root_dir=tmp_path)
        path2 = tmp_path / "manifest2.json"
        write_manifest_to_disk(manifest2, path2)
        invalid_file.unlink()  # Delete it

        report2_lenient = verify_manifest(path2, tmp_path, deep=False, lenient=True)
        assert report2_lenient.is_valid  # Valid in lenient mode

        # Invalid case in strict mode - missing file should fail
        report2_strict = verify_manifest(path2, tmp_path, deep=False, lenient=False)
        assert not report2_strict.is_valid

    def test_report_includes_all_files_in_graph(self, tmp_path):
        """Deep verification report should mention all files in the dependency graph."""
        # Create 2-level chain
        primary_dir = tmp_path / "1-benchmarks"
        primary_dir.mkdir(parents=True)
        primary = primary_dir / "benchmark.json"
        primary.write_text("{}")

        intermediate_dir = tmp_path / "3-processed"
        intermediate_dir.mkdir(parents=True)
        intermediate = intermediate_dir / "routes.json"
        intermediate.write_text("{}")

        manifest1 = create_simple_manifest("ingest", [intermediate], [primary], root_dir=tmp_path)
        manifest1_path = intermediate_dir / "manifest.json"
        write_manifest_to_disk(manifest1, manifest1_path)

        final_dir = tmp_path / "4-scored"
        final_dir.mkdir(parents=True)
        final = final_dir / "scores.json"
        final.write_text("{}")

        manifest2 = create_simple_manifest("score", [final], [intermediate], root_dir=tmp_path)
        manifest2_path = final_dir / "manifest.json"
        write_manifest_to_disk(manifest2, manifest2_path)

        # Deep verify
        report = verify_manifest(manifest2_path, tmp_path, deep=True)

        # Report should reference all 3 data files in the path fields
        issue_paths = [str(e.path) for e in report.issues]
        # Check that the files appear in some issues
        assert any("benchmark.json" in p or "1-benchmarks" in p for p in issue_paths)
        assert any("routes.json" in p or "3-processed" in p for p in issue_paths)
        assert any("scores.json" in p or "4-scored" in p for p in issue_paths)
