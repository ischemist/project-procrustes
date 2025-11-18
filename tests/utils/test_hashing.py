import hashlib
from pathlib import Path

import pytest

from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Molecule, Route
from retrocast.typing import InchiKeyStr, SmilesStr
from retrocast.utils.hashing import (
    compute_routes_content_hash,
    generate_file_hash,
    generate_model_hash,
    generate_source_hash,
)


def test_generate_model_hash_is_deterministic() -> None:
    """Tests that the same model name always produces the same short hash."""
    hash1 = generate_model_hash("test-model-v1")
    hash2 = generate_model_hash("test-model-v1")
    assert hash1 == hash2


def test_generate_model_hash_is_sensitive() -> None:
    """Tests that different model names produce different short hashes."""
    hash1 = generate_model_hash("test-model-v1")
    hash2 = generate_model_hash("test-model-v2")
    assert hash1 != hash2


def test_generate_model_hash_has_correct_format_and_length() -> None:
    """Tests the prefix and truncated length of the model hash."""
    model_hash = generate_model_hash("any-model-name")
    prefix = "retrocasted-model-"
    assert model_hash.startswith(prefix)
    # The hash part should be exactly 8 characters long.
    assert len(model_hash) == len(prefix) + 8


def test_generate_source_hash_is_deterministic_and_order_invariant() -> None:
    """
    Tests that the source hash is deterministic and invariant to the order
    of file hashes.
    """
    model_name = "test-model"
    file_hashes_1 = ["hash_a", "hash_b", "hash_c"]
    file_hashes_2 = ["hash_c", "hash_a", "hash_b"]  # Same hashes, different order

    source_hash_1 = generate_source_hash(model_name, file_hashes_1)
    source_hash_2 = generate_source_hash(model_name, file_hashes_2)

    assert source_hash_1 == source_hash_2
    assert source_hash_1.startswith("retrocasted-source-")


def test_generate_source_hash_is_sensitive_to_model_name() -> None:
    """Tests that changing the model name changes the source hash."""
    model_name_1 = "model-a"
    model_name_2 = "model-b"
    file_hashes = ["hash_a", "hash_b", "hash_c"]

    source_hash_1 = generate_source_hash(model_name_1, file_hashes)
    source_hash_2 = generate_source_hash(model_name_2, file_hashes)

    assert source_hash_1 != source_hash_2


def test_generate_file_hash_is_correct(tmp_path: Path) -> None:
    """
    Tests that generate_file_hash correctly computes the sha256 of a file's content.
    """
    content = b"retrocast major is the best bear"
    expected_hash = hashlib.sha256(content).hexdigest()
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(content)
    calculated_hash = generate_file_hash(file_path)
    assert calculated_hash == expected_hash


def test_generate_file_hash_raises_exception_for_missing_file(tmp_path: Path) -> None:
    """Tests that our custom exception is raised if the file does not exist."""
    non_existent_path = tmp_path / "this_file_does_not_exist.txt"
    with pytest.raises(RetroCastException):
        generate_file_hash(non_existent_path)


# ==============================================================================
# compute_routes_content_hash Tests
# ==============================================================================


@pytest.fixture
def sample_routes() -> dict[str, list[Route]]:
    """Create sample routes for testing content hash."""
    mol1 = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))
    mol2 = Molecule(smiles=SmilesStr("c1ccccc1"), inchikey=InchiKeyStr("UHOVQNZJYSORNB-UHFFFAOYSA-N"))

    route1 = Route(target=mol1, rank=1)
    route2 = Route(target=mol2, rank=1)
    route3 = Route(target=mol2, rank=2)

    return {"target_1": [route1], "target_2": [route2, route3]}


def test_compute_routes_content_hash_is_deterministic(sample_routes: dict[str, list[Route]]) -> None:
    """Test that content hash is deterministic."""
    hash1 = compute_routes_content_hash(sample_routes)
    hash2 = compute_routes_content_hash(sample_routes)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_compute_routes_content_hash_order_agnostic() -> None:
    """Test that content hash is the same regardless of dict insertion order."""
    mol1 = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))
    mol2 = Molecule(smiles=SmilesStr("c1ccccc1"), inchikey=InchiKeyStr("UHOVQNZJYSORNB-UHFFFAOYSA-N"))

    route1 = Route(target=mol1, rank=1)
    route2 = Route(target=mol2, rank=1)

    # Different insertion order
    routes_order1 = {"target_a": [route1], "target_b": [route2]}
    routes_order2 = {"target_b": [route2], "target_a": [route1]}

    hash1 = compute_routes_content_hash(routes_order1)
    hash2 = compute_routes_content_hash(routes_order2)
    assert hash1 == hash2


def test_compute_routes_content_hash_different_routes_different_hash() -> None:
    """Test that different route data produces different hashes."""
    mol1 = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))
    mol2 = Molecule(smiles=SmilesStr("c1ccccc1"), inchikey=InchiKeyStr("UHOVQNZJYSORNB-UHFFFAOYSA-N"))

    routes1 = {"target": [Route(target=mol1, rank=1)]}
    routes2 = {"target": [Route(target=mol2, rank=1)]}

    hash1 = compute_routes_content_hash(routes1)
    hash2 = compute_routes_content_hash(routes2)
    assert hash1 != hash2


def test_compute_routes_content_hash_empty_dict() -> None:
    """Test content hash of empty routes dictionary."""
    hash1 = compute_routes_content_hash({})
    hash2 = compute_routes_content_hash({})
    assert hash1 == hash2
    assert len(hash1) == 64


def test_compute_routes_content_hash_sensitive_to_metadata() -> None:
    """Test that metadata changes affect the content hash."""
    mol = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))

    routes1 = {"target": [Route(target=mol, rank=1, metadata={"score": 0.9})]}
    routes2 = {"target": [Route(target=mol, rank=1, metadata={"score": 0.95})]}

    hash1 = compute_routes_content_hash(routes1)
    hash2 = compute_routes_content_hash(routes2)
    assert hash1 != hash2


def test_compute_routes_content_hash_sensitive_to_rank() -> None:
    """Test that route rank changes affect the content hash."""
    mol = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))

    routes1 = {"target": [Route(target=mol, rank=1)]}
    routes2 = {"target": [Route(target=mol, rank=2)]}

    hash1 = compute_routes_content_hash(routes1)
    hash2 = compute_routes_content_hash(routes2)
    assert hash1 != hash2


def test_compute_routes_content_hash_sorts_by_rank() -> None:
    """Test that routes are sorted by rank before hashing."""
    mol = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))

    route1 = Route(target=mol, rank=1)
    route2 = Route(target=mol, rank=2)

    # Different list order, same ranks
    routes_order1 = {"target": [route1, route2]}
    routes_order2 = {"target": [route2, route1]}

    hash1 = compute_routes_content_hash(routes_order1)
    hash2 = compute_routes_content_hash(routes_order2)
    assert hash1 == hash2
