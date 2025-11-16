from unittest.mock import patch

import pytest

from retrocast.domain.chem import canonicalize_smiles, get_heavy_atom_count, get_inchi_key, get_molecular_weight
from retrocast.exceptions import InvalidSmilesError, RetroCastException


def test_canonicalize_smiles_valid_non_canonical() -> None:
    """Tests that a valid, non-canonical SMILES is correctly canonicalized."""
    # Arrange
    non_canonical_smiles = "C(C)O"  # Ethanol
    expected_canonical = "CCO"

    # Act
    result = canonicalize_smiles(non_canonical_smiles)

    # Assert
    assert result == expected_canonical


def test_canonicalize_smiles_already_canonical() -> None:
    """Tests that an already-canonical SMILES string remains unchanged."""
    # Arrange
    canonical_smiles = "CCO"

    # Act
    result = canonicalize_smiles(canonical_smiles)

    # Assert
    assert result == canonical_smiles


def test_canonicalize_smiles_with_stereochemistry() -> None:
    """Tests that stereochemical information is preserved."""
    # Arrange
    chiral_smiles = "C[C@H](O)C(=O)O"  # (R)-Lactic acid
    # RDKit's canonical form for this might vary, but it should be consistent
    # and contain stereochemical markers. Let's find out what it is.
    expected_canonical = "C[C@H](O)C(=O)O"

    # Act
    result = canonicalize_smiles(chiral_smiles)

    # Assert
    assert result == expected_canonical


def test_canonicalize_smiles_invalid_raises_error() -> None:
    """
    Tests that passing a chemically invalid string raises InvalidSmilesError.
    """
    # Arrange
    invalid_smiles = "this is definitely not a valid smiles string C(C)O"

    # Act / Assert
    with pytest.raises(InvalidSmilesError) as exc_info:
        canonicalize_smiles(invalid_smiles)

    # Optional: check the exception message for clarity
    assert "Invalid SMILES string" in str(exc_info.value)


@pytest.mark.parametrize("bad_input", ["", None, 123])
def test_canonicalize_smiles_bad_input_type_raises_error(bad_input) -> None:
    """
    Tests that non-string or empty inputs raise InvalidSmilesError.
    """
    # Arrange (done by parametrize)

    # Act / Assert
    with pytest.raises(InvalidSmilesError) as exc_info:
        canonicalize_smiles(bad_input)

    assert "SMILES input must be a non-empty string" in str(exc_info.value)


def test_get_inchi_key_happy_path() -> None:
    """tests that a simple smiles gives the correct, known inchikey."""
    # arrange
    smiles = "c1ccccc1"  # benzene
    expected_key = "UHOVQNZJYSORNB-UHFFFAOYSA-N".lower()

    # act
    result = get_inchi_key(smiles)

    # assert
    assert result.lower() == expected_key


def test_get_inchi_key_is_canonical() -> None:
    """tests that different valid smiles for the same molecule yield the same inchikey."""
    # arrange
    smiles1 = "CCO"  # ethanol
    smiles2 = "OCC"

    # act
    key1 = get_inchi_key(smiles1)
    key2 = get_inchi_key(smiles2)

    # assert
    assert key1 is not None
    assert key1 == key2


def test_get_inchi_key_handles_stereochemistry() -> None:
    """
    tests that stereoisomers produce different inchikeys, and are different
    from the racemic/unspecified version. this is a critical property.
    """
    # arrange
    d_alanine = "C[C@@H](C(=O)O)N"
    l_alanine = "C[C@H](C(=O)O)N"
    unspec_alanine = "CC(C(=O)O)N"

    # act
    d_key = get_inchi_key(d_alanine)
    l_key = get_inchi_key(l_alanine)
    unspec_key = get_inchi_key(unspec_alanine)

    # assert
    # all keys must be different
    assert d_key != l_key
    assert d_key != unspec_key
    assert l_key != unspec_key

    # the "fixed-h" and "stereo" layers should differ
    assert d_key.lower() == "QNAYBMKLOCPYGJ-REOHCLBHSA-N".lower()
    assert l_key.lower() == "QNAYBMKLOCPYGJ-UWTATZPHSA-N".lower()
    assert unspec_key.lower() == "QNAYBMKLOCPYGJ-UHFFFAOYSA-N".lower()


def test_get_inchi_key_invalid_smiles_raises_error() -> None:
    """tests that a malformed smiles raises our specific exception."""
    # arrange
    invalid_smiles = "C(C)C)C"  # mismatched parentheses

    # act / assert
    with pytest.raises(InvalidSmilesError) as exc_info:
        get_inchi_key(invalid_smiles)

    assert "invalid smiles string" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "bad_input",
    [
        "",
        None,
        42,
    ],
)
def test_get_inchi_key_bad_input_raises_error(bad_input) -> None:
    """tests that non-string or empty inputs raise our specific exception."""
    # arrange (from param)

    # act / assert
    with pytest.raises(InvalidSmilesError):
        get_inchi_key(bad_input)


@patch("retrocast.domain.chem.Chem.MolToSmiles")
def test_canonicalize_smiles_raises_ursa_exception_on_generic_error(mock_moltosmiles) -> None:
    """
    tests that a generic, unexpected rdkit error is wrapped in our RetroCastException.
    this covers the final `except` block.
    """
    # arrange: force the rdkit function to raise a generic error
    mock_moltosmiles.side_effect = RuntimeError("some esoteric rdkit failure")

    # act / assert
    with pytest.raises(RetroCastException) as exc_info:
        canonicalize_smiles("CCO")  # a valid smiles that will now fail

    assert "An unexpected error occurred during SMILES processing" in str(exc_info.value)


@patch("retrocast.domain.chem.Chem.MolToInchiKey")
def test_get_inchi_key_raises_ursa_exception_on_empty_result(mock_moltoinchikey) -> None:
    """
    tests that our guard for an empty inchikey from rdkit works.
    this covers the `if not inchi_key:` block.
    """
    # arrange: force rdkit to return an empty string
    mock_moltoinchikey.return_value = ""

    # act / assert
    with pytest.raises(RetroCastException) as exc_info:
        get_inchi_key("CCO")

    assert "produced an empty InChIKey" in str(exc_info.value)


@patch("retrocast.domain.chem.Chem.MolToInchiKey")
def test_get_inchi_key_raises_ursa_exception_on_generic_error(mock_moltoinchikey) -> None:
    """
    tests that a generic, unexpected rdkit error is wrapped in our RetroCastException.
    this covers the final `except` block in get_inchi_key.
    """
    # arrange
    mock_moltoinchikey.side_effect = RuntimeError("another esoteric rdkit failure")

    # act / assert
    with pytest.raises(RetroCastException) as exc_info:
        get_inchi_key("CCO")

    assert "An unexpected error occurred during InChIKey generation" in str(exc_info.value)


@pytest.mark.parametrize(
    "smiles,expected_count",
    [
        ("C", 1),  # methane
        ("CCO", 3),  # ethanol
        ("c1ccccc1", 6),  # benzene
        ("CC(=O)O", 4),  # acetic acid
        ("C[C@H](O)C(=O)O", 6),  # lactic acid
    ],
)
def test_get_heavy_atom_count(smiles: str, expected_count: int) -> None:
    """Tests that heavy atom count is correctly computed for various molecules."""
    result = get_heavy_atom_count(smiles)
    assert result == expected_count


@pytest.mark.parametrize(
    "smiles,expected_mw",
    [
        ("C", 16.031),  # methane
        ("CCO", 46.042),  # ethanol
        ("c1ccccc1", 78.047),  # benzene
        ("CC(=O)O", 60.021),  # acetic acid
        ("C[C@H](O)C(=O)O", 90.032),  # lactic acid
    ],
)
def test_get_molecular_weight(smiles: str, expected_mw: float) -> None:
    """Tests that molecular weight is correctly computed for various molecules."""
    result = get_molecular_weight(smiles)
    assert result == pytest.approx(expected_mw, rel=1e-3)
