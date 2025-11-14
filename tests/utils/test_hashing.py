import hashlib
from pathlib import Path

import pytest

from retrocast.exceptions import UrsaException
from retrocast.utils.hashing import (
    generate_file_hash,
    generate_model_hash,
    generate_molecule_hash,
    generate_source_hash,
)


def test_generate_molecule_hash_is_deterministic() -> None:
    """Tests that the same SMILES string always produces the same hash."""
    smiles1 = "CCO"
    smiles2 = "CCO"
    hash1 = generate_molecule_hash(smiles1)
    hash2 = generate_molecule_hash(smiles2)
    assert hash1 == hash2
    assert hash1.startswith("sha256-")


def test_generate_molecule_hash_is_sensitive() -> None:
    """Tests that different SMILES strings produce different hashes."""
    smiles1 = "CCO"  # Ethanol
    smiles2 = "COC"  # Dimethyl ether
    hash1 = generate_molecule_hash(smiles1)
    hash2 = generate_molecule_hash(smiles2)
    assert hash1 != hash2


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
    prefix = "ursa-model-"
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
    assert source_hash_1.startswith("ursa-source-")


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
    content = b"ursa major is the best bear"
    expected_hash = hashlib.sha256(content).hexdigest()
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(content)
    calculated_hash = generate_file_hash(file_path)
    assert calculated_hash == expected_hash


def test_generate_file_hash_raises_exception_for_missing_file(tmp_path: Path) -> None:
    """Tests that our custom exception is raised if the file does not exist."""
    non_existent_path = tmp_path / "this_file_does_not_exist.txt"
    with pytest.raises(UrsaException):
        generate_file_hash(non_existent_path)
