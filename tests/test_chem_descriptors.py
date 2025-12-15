from unittest.mock import patch

import pytest

from retrocast.chem import (
    get_chiral_center_count,
    get_heavy_atom_count,
    get_inchi_key,
    get_molecular_weight,
)
from retrocast.exceptions import InvalidSmilesError, RetroCastException

# ============================================================================
# Tests for get_heavy_atom_count
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_count",
    [
        ("C", 1),  # methane
        ("CCO", 3),  # ethanol
        ("c1ccccc1", 6),  # benzene
        ("C[C@H](O)C(=O)O", 6),  # lactic acid
    ],
)
def test_get_heavy_atom_count(smiles: str, expected_count: int) -> None:
    """Tests that heavy atom count is correctly computed."""
    result = get_heavy_atom_count(smiles)
    assert result == expected_count


@pytest.mark.unit
def test_get_heavy_atom_count_invalid_smiles_raises_error() -> None:
    """Tests that invalid SMILES raises InvalidSmilesError."""
    invalid_smiles = "C(C)C)C"
    with pytest.raises(InvalidSmilesError):
        get_heavy_atom_count(invalid_smiles)


@pytest.mark.unit
@patch("retrocast.chem.Chem.MolFromSmiles")
def test_get_heavy_atom_count_raises_exception_on_generic_error(mock_molfromsmiles) -> None:
    """Tests that generic errors are wrapped in RetroCastException."""
    mock_molfromsmiles.side_effect = RuntimeError("unexpected rdkit error")
    with pytest.raises(RetroCastException) as exc_info:
        get_heavy_atom_count("CCO")
    assert "An unexpected error occurred during HAC calculation" in str(exc_info.value)


# ============================================================================
# Tests for get_molecular_weight
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_mw",
    [
        ("C", 16.031),  # methane
        ("CCO", 46.042),  # ethanol
        ("c1ccccc1", 78.047),  # benzene
        ("C[C@H](O)C(=O)O", 90.032),  # lactic acid
    ],
)
def test_get_molecular_weight(smiles: str, expected_mw: float) -> None:
    """Tests that molecular weight is correctly computed."""
    result = get_molecular_weight(smiles)
    assert result == pytest.approx(expected_mw, rel=1e-3)


@pytest.mark.unit
def test_get_molecular_weight_invalid_smiles_raises_error() -> None:
    """Tests that invalid SMILES raises InvalidSmilesError."""
    invalid_smiles = "C(C)C)C"
    with pytest.raises(InvalidSmilesError):
        get_molecular_weight(invalid_smiles)


@pytest.mark.unit
@patch("retrocast.chem.rdMolDescriptors.CalcExactMolWt")
def test_get_molecular_weight_raises_exception_on_generic_error(mock_calc) -> None:
    """Tests that generic errors are wrapped in RetroCastException."""
    mock_calc.side_effect = RuntimeError("unexpected rdkit error")
    with pytest.raises(RetroCastException) as exc_info:
        get_molecular_weight("CCO")
    assert "An unexpected error occurred during MW calculation" in str(exc_info.value)


# ============================================================================
# Tests for get_chiral_center_count
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_count",
    [
        ("C", 0),  # methane - no chiral centers
        ("CCO", 0),  # ethanol - no chiral centers
        ("C[C@H](O)C(=O)O", 1),  # lactic acid - 1 chiral center
        ("C[C@H](O)[C@H](O)C", 2),  # 2,3-butanediol - 2 chiral centers
    ],
)
def test_get_chiral_center_count(smiles: str, expected_count: int) -> None:
    """Tests that chiral center count is correctly computed."""
    result = get_chiral_center_count(smiles)
    assert result == expected_count


@pytest.mark.unit
def test_get_chiral_center_count_invalid_smiles_raises_error() -> None:
    """Tests that invalid SMILES raises InvalidSmilesError."""
    invalid_smiles = "C(C)C)C"
    with pytest.raises(InvalidSmilesError):
        get_chiral_center_count(invalid_smiles)


@pytest.mark.unit
@patch("retrocast.chem.Chem.FindMolChiralCenters")
def test_get_chiral_center_count_raises_exception_on_generic_error(mock_find) -> None:
    """Tests that generic errors are wrapped in RetroCastException."""
    mock_find.side_effect = RuntimeError("unexpected rdkit error")
    with pytest.raises(RetroCastException) as exc_info:
        get_chiral_center_count("CCO")
    assert "An unexpected error occurred during chiral center count" in str(exc_info.value)


# ============================================================================
# Shared exception tests (covers all functions with one parametrized test)
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "func",
    [
        get_inchi_key,
        get_heavy_atom_count,
        get_molecular_weight,
        get_chiral_center_count,
    ],
)
@pytest.mark.parametrize("bad_input", ["", None, 123])
def test_all_functions_reject_bad_input(func, bad_input) -> None:
    """Tests that all chem functions reject non-string or empty inputs."""
    with pytest.raises(InvalidSmilesError) as exc_info:
        func(bad_input)
    assert "SMILES input must be a non-empty string" in str(exc_info.value)
