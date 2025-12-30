from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retrocast.chem import (
    NO_STEREO_PLACEHOLDER,
    InchiKeyLevel,
    canonicalize_smiles,
    get_chiral_center_count,
    get_heavy_atom_count,
    get_inchi_key,
    get_molecular_weight,
    reduce_inchikey,
)
from retrocast.exceptions import InvalidSmilesError, RetroCastException

# ============================================================================
# canonicalization
# ============================================================================


@pytest.mark.unit
def test_canonicalize_smiles_valid_non_canonical() -> None:
    assert canonicalize_smiles("C(C)O") == "CCO"


@pytest.mark.unit
def test_canonicalize_smiles_with_stereochemistry() -> None:
    s = "C[C@H](O)C(=O)O"
    assert canonicalize_smiles(s) == s


@pytest.mark.unit
def test_canonicalize_smiles_remove_mapping() -> None:
    assert canonicalize_smiles("[CH3:1][CH2:2][OH:3]", remove_mapping=True) == "CCO"


@pytest.mark.unit
@given(st.sampled_from(["CCO", "OCC", "C(C)O"]))
def test_canonicalize_is_idempotent(smiles: str) -> None:
    first = canonicalize_smiles(smiles)
    assert canonicalize_smiles(first) == first


@pytest.mark.unit
@given(
    st.sampled_from(
        [
            "C(C)O",  # non-canonical ethanol
            "C1=CC=CC=C1",  # non-canonical benzene
            "O[C@H](C)C(=O)O",  # non-canonical lactic
        ]
    )
)
def test_canonicalize_preserves_molecular_properties(smiles: str) -> None:
    """
    invariant: canonicalization changes representation, not chemistry.
    HAC, MW, and chiral centers must be identical pre/post canon.
    """
    canonical = canonicalize_smiles(smiles)

    assert get_heavy_atom_count(smiles) == get_heavy_atom_count(canonical)
    assert get_molecular_weight(smiles) == pytest.approx(get_molecular_weight(canonical))
    assert get_chiral_center_count(smiles) == get_chiral_center_count(canonical)


@pytest.mark.unit
@given(
    st.sampled_from(
        [
            ("CCO", "[CH3:1][CH2:2][OH:3]"),
            ("c1ccccc1", "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"),
        ]
    )
)
def test_remove_mapping_equivalence(smiles_pair: tuple[str, str]) -> None:
    """
    invariant: canonicalize(mapped, remove_mapping=True) should equal
    canonicalize(unmapped). ensures mapping removal doesn't corrupt structure.
    """
    unmapped, mapped = smiles_pair

    canon_unmapped = canonicalize_smiles(unmapped)
    canon_mapped_stripped = canonicalize_smiles(mapped, remove_mapping=True)

    assert canon_unmapped == canon_mapped_stripped


@pytest.mark.unit
@given(
    st.sampled_from(
        [
            "C[C@H](O)C(=O)O",
            "C[C@@H](O)C(=O)O",
            "C[C@H]([C@H](O)C)N",
        ]
    )
)
def test_canonicalize_isomeric_false_strips_stereo(smiles: str) -> None:
    """
    invariant: canonicalize(s, ignore_stereo=True) should produce SMILES
    without @/@@ markers. InChI key of result should match NO_STEREO.
    """
    non_isomeric = canonicalize_smiles(smiles, ignore_stereo=True)

    # no stereo markers in output
    assert "@" not in non_isomeric

    # InChI key should have NO_STEREO placeholder
    key = get_inchi_key(non_isomeric, InchiKeyLevel.FULL)
    assert NO_STEREO_PLACEHOLDER in key


# ============================================================================
# inchikey generation & levels
# ============================================================================


@pytest.mark.unit
def test_get_inchi_key_happy_path() -> None:
    # benzene
    assert get_inchi_key("c1ccccc1") == "UHOVQNZJYSORNB-UHFFFAOYSA-N"


@pytest.mark.unit
def test_get_inchi_key_levels() -> None:
    # R-lactic acid
    s = "C[C@H](O)C(=O)O"

    # full: has stereo layer
    full = get_inchi_key(s, InchiKeyLevel.FULL)
    assert "UHFFFAOYSA" not in full

    # no_stereo: stereo layer replaced by standard null
    ns = get_inchi_key(s, InchiKeyLevel.NO_STEREO)
    assert "UHFFFAOYSA" in ns
    assert len(ns) == 27

    # connectivity: truncated
    conn = get_inchi_key(s, InchiKeyLevel.CONNECTIVITY)
    assert len(conn) == 14
    assert "-" not in conn
    assert full.startswith(conn)


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene - planar, no stereo
        "CCO",  # ethanol - no chiral centers
        "CC(C)C",  # isobutane - symmetric
        "C1CCCCC1",  # cyclohexane - no defined stereo
    ],
)
def test_achiral_molecules_stereo_handling(smiles: str) -> None:
    """
    achiral molecules should produce NO_STEREO keys with the standard placeholder,
    even when generated with FULL level (because rdkit sees no stereo to encode).
    """
    full = get_inchi_key(smiles, InchiKeyLevel.FULL)
    no_stereo = get_inchi_key(smiles, InchiKeyLevel.NO_STEREO)

    # both should have the NO_STEREO_PLACEHOLDER
    assert NO_STEREO_PLACEHOLDER in full
    assert NO_STEREO_PLACEHOLDER in no_stereo

    # and they should be identical
    assert full == no_stereo


@pytest.mark.unit
@given(
    smiles=st.sampled_from(["C", "CCO", "c1ccccc1", "C[C@H](O)C"]),
    level=st.sampled_from([InchiKeyLevel.FULL, InchiKeyLevel.NO_STEREO]),
)
def test_inchikey_format_structure(smiles: str, level: InchiKeyLevel) -> None:
    """
    invariant: all 27-char InChI keys must have format:
    - 14 alphanumeric chars
    - hyphen
    - 10 alphanumeric chars (8 stereo + 2 from suffix)
    - hyphen
    - 1 char suffix
    """
    key = get_inchi_key(smiles, level=level)

    assert len(key) == 27
    parts = key.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 14
    assert len(parts[1]) == 10
    assert len(parts[2]) == 1
    assert all(c.isalnum() or c == "-" for c in key)


# ============================================================================
# normalization & consistency (the critical stuff)
# ============================================================================


@pytest.mark.unit
def test_reduce_inchikey_prevent_upscaling() -> None:
    """verifies we can't hallucinate stereo info from a skeleton."""
    conn_key = "UHOVQNZJYSORNB"

    # prevent 14 -> 27
    with pytest.raises(RetroCastException, match="cannot upscale"):
        reduce_inchikey(conn_key, InchiKeyLevel.FULL)

    with pytest.raises(RetroCastException, match="cannot upscale"):
        reduce_inchikey(conn_key, InchiKeyLevel.NO_STEREO)


@pytest.mark.unit
@given(
    st.sampled_from(
        [
            "C[C@H](O)C(=O)O",  # lactic
            "C[C@@H](C(=O)O)N",  # alanine
            "O[C@H](F)Cl",  # halogenated chaos
        ]
    )
)
def test_generation_reduction_consistency(smiles: str) -> None:
    """
    invariant: get_inchi(s, NO_STEREO) == reduce(get_inchi(s, FULL), NO_STEREO)

    ensures rdkit's '-SNon' flag aligns with our string surgery.
    """
    direct = get_inchi_key(smiles, level=InchiKeyLevel.NO_STEREO)
    full = get_inchi_key(smiles, level=InchiKeyLevel.FULL)
    reduced = reduce_inchikey(full, level=InchiKeyLevel.NO_STEREO)

    assert direct == reduced


@pytest.mark.unit
def test_stereo_saturation_collapse() -> None:
    """
    tests that all stereoisomers of a chiral chain collapse to a single
    NO_STEREO identifier. uses 2,3,4-pentanetriol topology.
    """
    isomers = [
        "C[C@H]([C@@H]([C@@H](O)C)O)O",
        "C[C@@H](C([C@@H](O)C)O)O",
        "C[C@H](C([C@H](O)C)O)O",
        "C[C@H]([C@H]([C@@H](O)C)O)O",
        "C[C@@H]([C@@H]([C@H](O)C)O)O",
        "C[C@@H]([C@H]([C@H](O)C)O)O",
    ]

    # 1. verify we actually have distinct input isomers (chem check)
    full_keys = {get_inchi_key(s, InchiKeyLevel.FULL) for s in isomers}
    # we expect multiple distinct full keys (some might be meso/dup, but >1)
    assert len(full_keys) > 1

    # 2. verify absolute collapse
    ns_keys = {get_inchi_key(s, InchiKeyLevel.NO_STEREO) for s in isomers}
    assert len(ns_keys) == 1

    # 3. verify structure of the collapse
    collapsed = ns_keys.pop()
    parts = collapsed.split("-")
    assert parts[0] == "JJAIIULJXXEFLV"  # shared connectivity
    assert parts[1] == NO_STEREO_PLACEHOLDER
    assert parts[2] == "N"


@pytest.mark.unit
@given(
    st.sampled_from(
        [
            "C[C@H](O)C(=O)O",
            "C[C@@H]([C@@H](O)C)N",
            "c1cc([C@H](O)F)ccc1",
        ]
    )
)
def test_inchikey_reduction_transitivity(smiles: str) -> None:
    """
    invariant: reduction is transitive. FULL->CONN should equal FULL->NS->CONN.
    ensures our string surgery doesn't create inconsistent states.
    """
    full = get_inchi_key(smiles, InchiKeyLevel.FULL)

    # path 1: direct
    conn_direct = reduce_inchikey(full, InchiKeyLevel.CONNECTIVITY)

    # path 2: via NO_STEREO
    ns = reduce_inchikey(full, InchiKeyLevel.NO_STEREO)
    conn_via_ns = reduce_inchikey(ns, InchiKeyLevel.CONNECTIVITY)

    assert conn_direct == conn_via_ns


# ============================================================================
# error handling
# ============================================================================
# cross-function consistency & integration tests
# ============================================================================


@pytest.mark.unit
def test_connectivity_divergence_implies_chirality() -> None:
    """
    logic test: if two molecules share connectivity but differ in FULL key,
    at least one must have chiral centers.
    """
    r_lactic = "C[C@H](O)C(=O)O"
    s_lactic = "C[C@@H](O)C(=O)O"

    # same connectivity
    assert get_inchi_key(r_lactic, InchiKeyLevel.CONNECTIVITY) == get_inchi_key(s_lactic, InchiKeyLevel.CONNECTIVITY)

    # different FULL
    assert get_inchi_key(r_lactic, InchiKeyLevel.FULL) != get_inchi_key(s_lactic, InchiKeyLevel.FULL)

    # therefore: chiral centers must exist
    assert get_chiral_center_count(r_lactic) > 0
    assert get_chiral_center_count(s_lactic) > 0


@pytest.mark.unit
def test_stereo_round_trip_invariant() -> None:
    """
    integration: stereoisomers that are canonicalized with ignore_stereo=True
    should produce molecules indistinguishable at the InChI NO_STEREO level.
    """
    isomers = [
        "C[C@H](O)C(=O)O",
        "C[C@@H](O)C(=O)O",
    ]

    # strip stereo via canonicalization
    stripped = [canonicalize_smiles(s, ignore_stereo=True) for s in isomers]

    # should be identical SMILES
    assert stripped[0] == stripped[1]

    # and identical NO_STEREO keys
    keys = [get_inchi_key(s, InchiKeyLevel.NO_STEREO) for s in stripped]
    assert keys[0] == keys[1]


# ============================================================================
# error handling
# ============================================================================


@pytest.mark.unit
def test_invalid_smiles_raises() -> None:
    with pytest.raises(InvalidSmilesError):
        get_inchi_key("not a smile")


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_key",
    [
        "TOOSHORT",  # < 14 chars
        "VALIDPART1-VALIDP",  # 2 parts (invalid)
        "PART1-PART2-PART3-PART4",  # 4 parts (invalid)
        "",  # empty
    ],
)
def test_reduce_inchikey_rejects_malformed_input(bad_key: str) -> None:
    """
    guard test: reduce_inchikey should reject malformed keys.
    only 14-char (connectivity) or 27-char (full) keys are valid.
    """
    with pytest.raises((ValueError, RetroCastException)):
        reduce_inchikey(bad_key, InchiKeyLevel.CONNECTIVITY)


@pytest.mark.unit
@patch("retrocast.chem.Chem.MolToInchiKey")
def test_rdkit_empty_result_guarded(mock_inchi) -> None:
    mock_inchi.return_value = ""
    with pytest.raises(RetroCastException, match="Empty InchiKey"):
        get_inchi_key("C")


@pytest.mark.unit
@patch("retrocast.chem.rdinchi.MolToInchi")
def test_no_stereo_uses_snon_flag(mock_moltoinchi) -> None:
    """
    contract test: verifies NO_STEREO level actually calls rdkit with '-SNon' option.
    """
    mock_moltoinchi.return_value = ("InChI=1S/C3H6O3/c1-2(4)3(5)6/h2,4H,1H3,(H,5,6)", 0, "", "", "")

    get_inchi_key("C[C@H](O)C(=O)O", level=InchiKeyLevel.NO_STEREO)

    # verify it was called with the -SNon option
    mock_moltoinchi.assert_called_once()
    args, kwargs = mock_moltoinchi.call_args
    assert kwargs.get("options") == "-SNon"
