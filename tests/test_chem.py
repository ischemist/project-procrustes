from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retrocast.chem import (
    InchiKeyLevel,
    canonicalize_smiles,
    get_chiral_center_count,
    get_heavy_atom_count,
    get_inchi_key,
    get_molecular_weight,
    normalize_inchikey,
)
from retrocast.exceptions import InvalidSmilesError, RetroCastException

# ============================================================================
# Tests for canonicalize_smiles
# ============================================================================


@pytest.mark.unit
def test_canonicalize_smiles_valid_non_canonical() -> None:
    """Tests that a valid, non-canonical SMILES is correctly canonicalized."""
    non_canonical_smiles = "C(C)O"  # Ethanol
    expected_canonical = "CCO"
    result = canonicalize_smiles(non_canonical_smiles)
    assert result == expected_canonical


@pytest.mark.unit
def test_canonicalize_smiles_with_stereochemistry() -> None:
    """Tests that stereochemical information is preserved."""
    chiral_smiles = "C[C@H](O)C(=O)O"  # (R)-Lactic acid
    expected_canonical = "C[C@H](O)C(=O)O"
    result = canonicalize_smiles(chiral_smiles)
    assert result == expected_canonical


@pytest.mark.unit
def test_canonicalize_smiles_remove_mapping() -> None:
    """Tests that atom mapping is removed when remove_mapping=True (covers GetAtoms loop)."""
    mapped_smiles = "[CH3:1][CH2:2][OH:3]"  # Ethanol with atom mapping
    expected = "CCO"  # Canonical ethanol without mapping
    result = canonicalize_smiles(mapped_smiles, remove_mapping=True)
    assert result == expected
    assert ":" not in result


@pytest.mark.unit
def test_canonicalize_smiles_invalid_raises_error() -> None:
    """Tests that passing a chemically invalid string raises InvalidSmilesError."""
    invalid_smiles = "this is definitely not a valid smiles string"
    with pytest.raises(InvalidSmilesError) as exc_info:
        canonicalize_smiles(invalid_smiles)
    assert "Invalid SMILES string" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("bad_input", ["", None, 123])
def test_canonicalize_smiles_bad_input_type_raises_error(bad_input) -> None:
    """Tests that non-string or empty inputs raise InvalidSmilesError."""
    with pytest.raises(InvalidSmilesError) as exc_info:
        canonicalize_smiles(bad_input)
    assert "SMILES input must be a non-empty string" in str(exc_info.value)


@pytest.mark.unit
@patch("retrocast.chem.Chem.MolToSmiles")
def test_canonicalize_smiles_raises_retrocast_exception_on_generic_error(mock_moltosmiles) -> None:
    """Tests that a generic, unexpected rdkit error is wrapped in RetroCastException."""
    mock_moltosmiles.side_effect = RuntimeError("some esoteric rdkit failure")
    with pytest.raises(RetroCastException) as exc_info:
        canonicalize_smiles("CCO")
    assert "An unexpected error occurred during SMILES processing" in str(exc_info.value)


# ============================================================================
# Tests for get_inchi_key
# ============================================================================


@pytest.mark.unit
def test_get_inchi_key_happy_path() -> None:
    """Tests that a simple smiles gives the correct, known inchikey."""
    smiles = "c1ccccc1"  # benzene
    expected_key = "UHOVQNZJYSORNB-UHFFFAOYSA-N".lower()
    result = get_inchi_key(smiles)
    assert result.lower() == expected_key


@pytest.mark.unit
def test_get_inchi_key_handles_stereochemistry() -> None:
    """Tests that stereoisomers produce different inchikeys."""
    d_alanine = "C[C@@H](C(=O)O)N"
    l_alanine = "C[C@H](C(=O)O)N"
    unspec_alanine = "CC(C(=O)O)N"

    d_key = get_inchi_key(d_alanine)
    l_key = get_inchi_key(l_alanine)
    unspec_key = get_inchi_key(unspec_alanine)

    # All keys must be different
    assert d_key != l_key
    assert d_key != unspec_key
    assert l_key != unspec_key


@pytest.mark.unit
def test_get_inchi_key_invalid_smiles_raises_error() -> None:
    """Tests that a malformed smiles raises InvalidSmilesError."""
    invalid_smiles = "C(C)C)C"  # mismatched parentheses
    with pytest.raises(InvalidSmilesError) as exc_info:
        get_inchi_key(invalid_smiles)
    assert "invalid smiles string" in str(exc_info.value).lower()


@pytest.mark.unit
@patch("retrocast.chem.Chem.MolToInchiKey")
def test_get_inchi_key_raises_retrocast_exception_on_empty_result(mock_moltoinchikey) -> None:
    """Tests that our guard for an empty inchikey from rdkit works."""
    mock_moltoinchikey.return_value = ""
    with pytest.raises(RetroCastException) as exc_info:
        get_inchi_key("CCO")
    assert "produced an empty InChIKey" in str(exc_info.value)


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


# ============================================================================
# Hypothesis-based property tests (PBT)
# ============================================================================


@pytest.mark.unit
@given(st.sampled_from(["CCO", "OCC", "C(C)O"]))
def test_canonicalize_smiles_is_idempotent(smiles: str) -> None:
    """Property test: canonicalizing twice should give the same result."""
    canonical_once = canonicalize_smiles(smiles)
    canonical_twice = canonicalize_smiles(canonical_once)
    assert canonical_once == canonical_twice


@pytest.mark.unit
@given(st.sampled_from([("CCO", "OCC"), ("c1ccccc1", "C1=CC=CC=C1")]))
def test_equivalent_smiles_canonicalize_identically(smiles_pair: tuple[str, str]) -> None:
    """Property test: equivalent molecules should canonicalize to the same SMILES."""
    smiles1, smiles2 = smiles_pair
    assert canonicalize_smiles(smiles1) == canonicalize_smiles(smiles2)


@pytest.mark.unit
@given(st.sampled_from(["CCO", "c1ccccc1", "CC(C)C"]))
def test_get_inchi_key_is_deterministic(smiles: str) -> None:
    """Property test: InChIKey should be deterministic and well-formed."""
    key1 = get_inchi_key(smiles)
    key2 = get_inchi_key(smiles)
    assert key1 == key2
    assert len(key1) == 27  # InChIKey format: 14-10-X
    assert key1.count("-") == 2


# ============================================================================
# Tests for InChI Key Levels and Normalization
# ============================================================================


@pytest.mark.unit
def test_get_inchi_key_no_stereo_level() -> None:
    """Tests that NO_STEREO level produces keys without stereochemistry info."""
    # Two stereoisomers of the same molecule
    r_lactic = "C[C@H](O)C(=O)O"  # R-lactic acid
    s_lactic = "C[C@@H](O)C(=O)O"  # S-lactic acid

    # Full keys should be different (stereochemistry encoded)
    full_r = get_inchi_key(r_lactic, level=InchiKeyLevel.FULL)
    full_s = get_inchi_key(s_lactic, level=InchiKeyLevel.FULL)
    assert full_r != full_s

    # NO_STEREO keys should be identical (stereochemistry ignored at generation)
    no_stereo_r = get_inchi_key(r_lactic, level=InchiKeyLevel.NO_STEREO)
    no_stereo_s = get_inchi_key(s_lactic, level=InchiKeyLevel.NO_STEREO)
    assert no_stereo_r == no_stereo_s

    # NO_STEREO should still produce a valid 27-char InChI key
    assert len(no_stereo_r) == 27
    assert no_stereo_r.count("-") == 2


@pytest.mark.unit
def test_get_inchi_key_connectivity_level() -> None:
    """Tests that CONNECTIVITY level returns only the first 14 characters."""
    smiles = "C[C@H](O)C(=O)O"  # Lactic acid
    conn_key = get_inchi_key(smiles, level=InchiKeyLevel.CONNECTIVITY)

    # Should be exactly 14 characters (connectivity block only)
    assert len(conn_key) == 14
    assert "-" not in conn_key

    # Should match the first block of the full key
    full_key = get_inchi_key(smiles, level=InchiKeyLevel.FULL)
    assert full_key.startswith(conn_key)


@pytest.mark.unit
def test_normalize_inchikey_no_stereo() -> None:
    """Tests that normalize_inchikey with NO_STEREO replaces stereo with standard placeholder."""
    full_key = "JVTAAEKCZFNVCJ-REOHCLBHSA-N"  # Example InChI key

    normalized = normalize_inchikey(full_key, InchiKeyLevel.NO_STEREO)

    # Should still have 3 blocks with standard no-stereo placeholder
    parts = normalized.split("-")
    assert len(parts) == 3
    assert parts[0] == "JVTAAEKCZFNVCJ"  # Connectivity preserved
    assert parts[1] == "UHFFFAOYSA"  # Standard no-stereo placeholder
    assert parts[2] == "N"  # Protonation suffix preserved
    assert len(normalized) == 27  # Standard InChI key length


@pytest.mark.unit
def test_normalize_inchikey_connectivity() -> None:
    """Tests that normalize_inchikey with CONNECTIVITY returns only first block."""
    full_key = "JVTAAEKCZFNVCJ-REOHCLBHSA-N"

    normalized = normalize_inchikey(full_key, InchiKeyLevel.CONNECTIVITY)

    assert normalized == "JVTAAEKCZFNVCJ"
    assert len(normalized) == 14


@pytest.mark.unit
def test_normalize_inchikey_full() -> None:
    """Tests that normalize_inchikey with FULL returns key unchanged."""
    full_key = "JVTAAEKCZFNVCJ-REOHCLBHSA-N"

    normalized = normalize_inchikey(full_key, InchiKeyLevel.FULL)

    assert normalized == full_key


@pytest.mark.unit
def test_stereoisomers_match_with_normalize() -> None:
    """Integration test: stereoisomers should match when using NO_STEREO normalization."""
    r_alanine = "C[C@@H](C(=O)O)N"  # D-alanine
    s_alanine = "C[C@H](C(=O)O)N"  # L-alanine

    # Get full keys (should be different)
    key_r = get_inchi_key(r_alanine)
    key_s = get_inchi_key(s_alanine)
    assert key_r != key_s

    # Normalize to NO_STEREO (should match)
    assert normalize_inchikey(key_r, InchiKeyLevel.NO_STEREO) == normalize_inchikey(key_s, InchiKeyLevel.NO_STEREO)


@pytest.mark.unit
def test_inchikey_level_enum_values() -> None:
    """Tests that InchiKeyLevel enum has expected string values."""
    assert InchiKeyLevel.FULL.value == "full"
    assert InchiKeyLevel.NO_STEREO.value == "no_stereo"
    assert InchiKeyLevel.CONNECTIVITY.value == "connectivity"
