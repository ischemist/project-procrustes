"""Tests for the retrocast.curation module."""

import json
from pathlib import Path

import pytest

from retrocast.curation import create_manifest
from retrocast.io import save_json_gz
from retrocast.schemas import Molecule, Route
from retrocast.typing import InchiKeyStr, SmilesStr


@pytest.fixture
def sample_routes() -> dict[str, list[Route]]:
    """Create sample routes for testing."""
    mol1 = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))
    mol2 = Molecule(smiles=SmilesStr("c1ccccc1"), inchikey=InchiKeyStr("UHOVQNZJYSORNB-UHFFFAOYSA-N"))

    route1 = Route(target=mol1, rank=1)
    route2 = Route(target=mol2, rank=1)
    route3 = Route(target=mol2, rank=2)

    return {"target_1": [route1], "target_2": [route2, route3]}


@pytest.fixture
def source_and_output_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create source and output files for manifest testing."""
    source_file = tmp_path / "source.json.gz"
    output_file = tmp_path / "output.json.gz"

    # Create source file with some content
    save_json_gz({"raw": "data"}, source_file)

    # Create output file with some content
    save_json_gz({"processed": "routes"}, output_file)

    return source_file, output_file


def test_create_manifest_basic_structure(
    tmp_path: Path, sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that manifest contains all required fields."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test_dataset", source_file, output_file, sample_routes)

    assert "dataset" in manifest
    assert "source_file" in manifest
    assert "source_file_hash" in manifest
    assert "output_file" in manifest
    assert "output_file_hash" in manifest
    assert "output_content_hash" in manifest
    assert "statistics" in manifest
    assert "timestamp" in manifest
    assert "retrocast_version" in manifest


def test_create_manifest_dataset_name(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that dataset name is correctly stored."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("n1-routes", source_file, output_file, sample_routes)

    assert manifest["dataset"] == "n1-routes"


def test_create_manifest_file_names(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that file names (not paths) are stored."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    assert manifest["source_file"] == "source.json.gz"
    assert manifest["output_file"] == "output.json.gz"


def test_create_manifest_hashes_are_sha256(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that file hashes are valid SHA256 hex strings."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    # SHA256 produces 64 character hex string
    assert len(manifest["source_file_hash"]) == 64
    assert len(manifest["output_file_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in manifest["source_file_hash"])
    assert all(c in "0123456789abcdef" for c in manifest["output_file_hash"])


def test_create_manifest_content_hash_exists(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that content hash is generated."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    assert isinstance(manifest["output_content_hash"], str)
    assert len(manifest["output_content_hash"]) > 0


def test_create_manifest_default_statistics(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that default statistics are computed when not provided."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    stats = manifest["statistics"]
    assert stats["n_targets"] == 2  # target_1 and target_2
    assert stats["n_routes_saved"] == 3  # 1 route for target_1, 2 for target_2


def test_create_manifest_custom_statistics(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that custom statistics override defaults."""
    source_file, output_file = source_and_output_files

    custom_stats = {
        "n_routes_source": 100,
        "n_routes_saved": 95,
        "custom_metric": "value",
    }

    manifest = create_manifest("test", source_file, output_file, sample_routes, statistics=custom_stats)

    assert manifest["statistics"] == custom_stats
    assert "n_targets" not in manifest["statistics"]


def test_create_manifest_timestamp_format(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that timestamp is in ISO format."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    # ISO format should contain 'T' and end with timezone info
    timestamp = manifest["timestamp"]
    assert "T" in timestamp
    assert "+" in timestamp or "Z" in timestamp or timestamp.endswith("+00:00")


def test_create_manifest_retrocast_version(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that retrocast version is included."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    assert isinstance(manifest["retrocast_version"], str)
    assert len(manifest["retrocast_version"]) > 0


def test_create_manifest_empty_routes(source_and_output_files: tuple[Path, Path]) -> None:
    """Test manifest creation with empty routes dictionary."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("empty", source_file, output_file, {})

    assert manifest["statistics"]["n_targets"] == 0
    assert manifest["statistics"]["n_routes_saved"] == 0


def test_create_manifest_is_json_serializable(
    sample_routes: dict[str, list[Route]], source_and_output_files: tuple[Path, Path]
) -> None:
    """Test that the manifest can be serialized to JSON."""
    source_file, output_file = source_and_output_files

    manifest = create_manifest("test", source_file, output_file, sample_routes)

    # Should not raise
    json_str = json.dumps(manifest)
    assert isinstance(json_str, str)

    # Should round-trip correctly
    loaded = json.loads(json_str)
    assert loaded == manifest
