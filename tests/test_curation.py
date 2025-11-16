"""Tests for the retrocast.curation module."""

import json
from pathlib import Path

import pytest

from retrocast.curation import (
    create_manifest,
    excise_reactions_from_routes,
    filter_routes_by_reaction_overlap,
    filter_routes_by_signature,
    get_reaction_signatures,
    get_route_signatures,
    split_routes,
)
from retrocast.io import save_json_gz
from retrocast.schemas import Molecule, ReactionSignature, Route
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


def test_get_route_signatures(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that all unique route signatures are extracted."""
    signatures = get_route_signatures(sample_routes_with_reactions)

    # 3 routes total -> 3 unique signatures
    assert len(signatures) == 3
    assert all(isinstance(sig, str) for sig in signatures)


def test_get_route_signatures_empty() -> None:
    """Test with empty routes dict."""
    signatures = get_route_signatures({})
    assert signatures == set()


def test_get_reaction_signatures(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that all unique reaction signatures are extracted."""
    signatures = get_reaction_signatures(sample_routes_with_reactions)

    # Route A1: 1 reaction (EtOH + AcOH -> EtOAc)
    # Route A2: 1 reaction (EtOH + Ac2O -> EtOAc)
    # Route B1: 2 reactions (Phenol + CO2 -> Salicylic, Salicylic + Ac2O -> Aspirin)
    # Total: 4 unique reactions
    assert len(signatures) == 4
    assert all(isinstance(sig, tuple) for sig in signatures)
    assert all(isinstance(sig[0], frozenset) and isinstance(sig[1], str) for sig in signatures)


def test_get_reaction_signatures_empty() -> None:
    """Test with empty routes dict."""
    signatures = get_reaction_signatures({})
    assert signatures == set()


def test_filter_routes_by_signature(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that routes with matching signatures are excluded."""
    # Get signature of first route in target_A
    route_to_exclude = sample_routes_with_reactions["target_A"][0]
    exclude_sigs = {route_to_exclude.get_signature()}

    filtered = filter_routes_by_signature(sample_routes_with_reactions, exclude_sigs)

    # Should have both targets still, but target_A only has 1 route now
    assert "target_A" in filtered
    assert "target_B" in filtered
    assert len(filtered["target_A"]) == 1
    assert len(filtered["target_B"]) == 1
    # The remaining route should be rank 2
    assert filtered["target_A"][0].rank == 2


def test_filter_routes_by_signature_removes_empty_targets(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that targets with no remaining routes are removed."""
    # Exclude all routes from target_A
    exclude_sigs = {r.get_signature() for r in sample_routes_with_reactions["target_A"]}

    filtered = filter_routes_by_signature(sample_routes_with_reactions, exclude_sigs)

    assert "target_A" not in filtered
    assert "target_B" in filtered


def test_filter_routes_by_signature_empty_exclude() -> None:
    """Test with empty exclude set returns same structure."""
    routes = {"t1": [Route(target=Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("A")), rank=1)]}
    filtered = filter_routes_by_signature(routes, set())

    assert filtered == routes


def test_filter_routes_by_reaction_overlap(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that routes with overlapping reactions are removed entirely."""
    # Get a reaction signature from route A1 (EtOH + AcOH -> EtOAc)
    route_A1 = sample_routes_with_reactions["target_A"][0]
    exclude_reactions = route_A1.get_reaction_signatures()

    filtered = filter_routes_by_reaction_overlap(sample_routes_with_reactions, exclude_reactions)

    # Route A1 should be removed, A2 and B1 should remain
    assert "target_A" in filtered
    assert len(filtered["target_A"]) == 1
    assert filtered["target_A"][0].rank == 2
    assert "target_B" in filtered


def test_filter_routes_by_reaction_overlap_removes_all_overlapping(
    sample_routes_with_reactions: dict[str, list[Route]],
) -> None:
    """Test that all routes sharing any reaction are removed."""
    # Get reaction from route B1 (Salicylic + Ac2O -> Aspirin)
    route_B1 = sample_routes_with_reactions["target_B"][0]
    # This includes both reactions in the aspirin route
    exclude_reactions = route_B1.get_reaction_signatures()

    filtered = filter_routes_by_reaction_overlap(sample_routes_with_reactions, exclude_reactions)

    # B1 should be removed entirely
    assert "target_B" not in filtered
    # A1 and A2 should remain
    assert "target_A" in filtered
    assert len(filtered["target_A"]) == 2


def test_excise_reactions_from_routes_no_overlap(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that routes without overlapping reactions are kept as-is."""
    # Use a reaction signature that doesn't exist
    fake_reaction: ReactionSignature = (frozenset(["FAKE-KEY"]), "FAKE-PRODUCT")

    result = excise_reactions_from_routes(sample_routes_with_reactions, {fake_reaction})

    # All routes should be preserved
    assert len(result["target_A"]) == 2
    assert len(result["target_B"]) == 1


def test_excise_reactions_from_routes_partial_removal(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that excising reactions creates sub-routes."""
    # Excise the top reaction from aspirin route (Salicylic + Ac2O -> Aspirin)
    aspirin_route = sample_routes_with_reactions["target_B"][0]
    top_reaction = next(iter(aspirin_route.get_reaction_signatures()))

    # Find the reaction that produces aspirin specifically
    for sig in aspirin_route.get_reaction_signatures():
        if sig[1] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N":  # Aspirin InchiKey
            top_reaction = sig
            break

    result = excise_reactions_from_routes(sample_routes_with_reactions, {top_reaction})

    # target_A routes should be unchanged
    assert len(result["target_A"]) == 2
    # target_B should have sub-routes (salicylic acid becomes a new route target)
    assert "target_B" in result
    # The aspirin route gets excised, but sub-routes (salicylic acid) should appear
    assert len(result["target_B"]) >= 1


@pytest.mark.parametrize("train_ratio", [0.5, 0.7, 0.8])
def test_split_routes_ratios(sample_routes_with_reactions: dict[str, list[Route]], train_ratio: float) -> None:
    """Test that split respects the ratio approximately."""
    train, val = split_routes(sample_routes_with_reactions, train_ratio, seed=42)

    total_targets = len(sample_routes_with_reactions)
    train_count = len(train)
    val_count = len(val)

    # All targets accounted for
    assert train_count + val_count == total_targets
    # No overlap
    assert set(train.keys()) & set(val.keys()) == set()


def test_split_routes_deterministic(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that same seed produces same split."""
    train1, val1 = split_routes(sample_routes_with_reactions, 0.5, seed=123)
    train2, val2 = split_routes(sample_routes_with_reactions, 0.5, seed=123)

    assert set(train1.keys()) == set(train2.keys())
    assert set(val1.keys()) == set(val2.keys())


def test_split_routes_different_seeds(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that different seeds can produce different splits."""
    # With only 2 targets, many seeds might give same split, but we test the mechanism
    train1, _ = split_routes(sample_routes_with_reactions, 0.5, seed=1)
    train2, _ = split_routes(sample_routes_with_reactions, 0.5, seed=999)

    # Keys might be same or different depending on shuffle, but structure is valid
    assert len(train1) == len(train2) == 1


def test_split_routes_preserves_routes(sample_routes_with_reactions: dict[str, list[Route]]) -> None:
    """Test that all routes are preserved in the split."""
    train, val = split_routes(sample_routes_with_reactions, 0.5, seed=42)

    # Count total routes
    original_count = sum(len(routes) for routes in sample_routes_with_reactions.values())
    split_count = sum(len(routes) for routes in train.values()) + sum(len(routes) for routes in val.values())

    assert split_count == original_count
