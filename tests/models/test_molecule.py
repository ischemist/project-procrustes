"""Unit tests for Molecule and TargetInput classes."""

from typing import Any

import pytest
from pydantic import ValidationError

from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Molecule, ReactionStep, TargetInput
from retrocast.typing import InchiKeyStr, SmilesStr

# ==============================================================================
# TargetInput Tests
# ==============================================================================


@pytest.mark.unit
class TestTargetInput:
    """Tests for the TargetInput class."""

    def test_basic_instantiation(self):
        """Test creating a TargetInput with valid id and SMILES."""
        target = TargetInput(id="test-mol-1", smiles=SmilesStr("CCO"))
        assert target.id == "test-mol-1"
        assert target.smiles == "CCO"

    def test_with_canonical_smiles(self):
        """Test TargetInput with canonicalized SMILES."""
        smiles = canonicalize_smiles("OCC")  # Should canonicalize to CCO
        target = TargetInput(id="ethanol", smiles=smiles)
        assert target.id == "ethanol"
        assert target.smiles == "CCO"

    def test_missing_id_raises_validation_error(self):
        """Test that missing id field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TargetInput(smiles=SmilesStr("CCO"))  # type: ignore
        assert "id" in str(exc_info.value)

    def test_missing_smiles_raises_validation_error(self):
        """Test that missing smiles field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TargetInput(id="test-mol")  # type: ignore
        assert "smiles" in str(exc_info.value)

    def test_empty_id(self):
        """Test that empty id is allowed (validation is just type checking)."""
        target = TargetInput(id="", smiles=SmilesStr("CCO"))
        assert target.id == ""

    def test_with_pharma_routes_examples(self, pharma_routes_data: dict[str, Any]):
        """Test TargetInput creation with pharma routes data."""
        # Vonoprazan
        vonoprazan_smiles = canonicalize_smiles(pharma_routes_data["vonoprazan-1"]["smiles"])
        target1 = TargetInput(id="vonoprazan-1", smiles=vonoprazan_smiles)
        assert target1.id == "vonoprazan-1"
        assert len(target1.smiles) > 0

        # Mitapivat
        mitapivat_smiles = canonicalize_smiles(pharma_routes_data["mitapivat-1"]["smiles"])
        target2 = TargetInput(id="mitapivat-1", smiles=mitapivat_smiles)
        assert target2.id == "mitapivat-1"
        assert len(target2.smiles) > 0


# ==============================================================================
# Molecule Tests
# ==============================================================================


@pytest.mark.unit
class TestMolecule:
    """Tests for the Molecule class."""

    def test_basic_instantiation(self):
        """Test creating a basic leaf molecule."""
        mol = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        assert mol.smiles == "CCO"
        assert mol.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        assert mol.synthesis_step is None
        assert mol.metadata == {}

    def test_is_leaf_property_true(self):
        """Test is_leaf returns True for molecule without synthesis_step."""
        mol = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        assert mol.is_leaf is True

    def test_is_leaf_property_false(self):
        """Test is_leaf returns False for molecule with synthesis_step."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant])
        product = Molecule(
            smiles=SmilesStr("CCOC"),
            inchikey=InchiKeyStr("KFZMGEQAYNKOFK-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        assert product.is_leaf is False

    def test_get_leaves_single_leaf(self):
        """Test get_leaves returns self for leaf molecule."""
        mol = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        leaves = mol.get_leaves()
        assert leaves == {mol}
        assert len(leaves) == 1

    def test_get_leaves_one_synthesis_step(self):
        """Test get_leaves with one synthesis step."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant1, reactant2])
        product = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        leaves = product.get_leaves()
        assert leaves == {reactant1, reactant2}
        assert len(leaves) == 2

    def test_get_leaves_deep_tree(self):
        """Test get_leaves with deep synthesis tree."""
        # Create leaf molecules
        leaf1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"))
        leaf2 = Molecule(smiles=SmilesStr("O"), inchikey=InchiKeyStr("XLYOFNOQVPJJNP-UHFFFAOYSA-M"))
        leaf3 = Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("QGZKDVFQNNGYKY-UHFFFAOYSA-N"))

        # Build intermediate level
        intermediate1 = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf1, leaf2]),
        )

        # Build top level
        product = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("FAKE-INCHIKEY-1"),
            synthesis_step=ReactionStep(reactants=[intermediate1, leaf3]),
        )

        leaves = product.get_leaves()
        assert leaves == {leaf1, leaf2, leaf3}
        assert len(leaves) == 3

    def test_get_leaves_deduplication(self):
        """Test that get_leaves deduplicates molecules with same InChIKey."""
        # Same molecule used twice
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant, reactant])  # Same molecule twice
        product = Molecule(
            smiles=SmilesStr("CCOCCO"),
            inchikey=InchiKeyStr("FAKE-INCHIKEY-2"),
            synthesis_step=step,
        )
        leaves = product.get_leaves()
        assert len(leaves) == 1
        assert reactant in leaves

    def test_molecule_hash(self):
        """Test that molecule hash is based on InChIKey."""
        mol1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        mol2 = Molecule(
            smiles=SmilesStr("OCC"),  # Different SMILES, same molecule
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        assert hash(mol1) == hash(mol2)

    def test_molecule_equality(self):
        """Test molecule equality based on InChIKey."""
        mol1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        mol2 = Molecule(
            smiles=SmilesStr("OCC"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        mol3 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        assert mol1 == mol2
        assert mol1 != mol3
        assert mol2 != mol3

    def test_molecules_in_set(self):
        """Test that molecules can be added to sets and deduplicated."""
        mol1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        mol2 = Molecule(
            smiles=SmilesStr("OCC"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        mol3 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        molecule_set = {mol1, mol2, mol3}
        assert len(molecule_set) == 2  # mol1 and mol2 are the same

    def test_metadata_handling(self):
        """Test that custom metadata can be stored."""
        metadata = {"score": 0.95, "template_id": "template_123"}
        mol = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
            metadata=metadata,
        )
        assert mol.metadata == metadata
        assert mol.metadata["score"] == 0.95
