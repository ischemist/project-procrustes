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
from retrocast.workflow.verify import verify_manifest

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
        f.write(manifest.model_dump_json(indent=2, exclude_none=True))


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

    def test_manifest_paths_can_be_relative_to_project_root_when_verifying_data_dir(self, tmp_path):
        project_root = tmp_path
        data_dir = project_root / "data" / "retrocast"
        release_dir = data_dir / "releases" / "tiny"
        release_dir.mkdir(parents=True)
        output = release_dir / "all.jsonl.gz"
        output.write_text("{}\n", encoding="utf-8")

        manifest = create_simple_manifest("release", [output], root_dir=project_root)
        manifest_path = release_dir / "manifest.json"
        write_manifest_to_disk(manifest, manifest_path)

        report = verify_manifest(manifest_path, data_dir, deep=False)

        assert report.is_valid
        assert not any(issue.message == "File is MISSING from disk." for issue in report.issues)

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
