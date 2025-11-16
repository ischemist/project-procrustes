"""Unit tests for ReactionStep class."""

from retrocast.schemas import Molecule, ReactionStep
from retrocast.typing import InchiKeyStr, ReactionSmilesStr, SmilesStr

# ==============================================================================
# ReactionStep Tests
# ==============================================================================


class TestReactionStep:
    """Tests for the ReactionStep class."""

    def test_basic_instantiation(self):
        """Test creating a basic ReactionStep."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant1, reactant2])
        assert len(step.reactants) == 2
        assert step.mapped_smiles is None
        assert step.template is None
        assert step.reagents is None
        assert step.solvents is None
        assert step.metadata == {}

    def test_with_all_optional_fields(self):
        """Test ReactionStep with all optional fields populated."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(
            reactants=[reactant],
            mapped_smiles=ReactionSmilesStr("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3]"),
            template="[C:1][OH:2]>>[C:1][O:2]",
            reagents=[SmilesStr("O=S(=O)(Cl)Cl")],  # Thionyl chloride
            solvents=[SmilesStr("ClCCl")],  # Dichloromethane
            metadata={"patent_id": "US1234567"},
        )
        assert step.mapped_smiles == "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3]"
        assert step.template == "[C:1][OH:2]>>[C:1][O:2]"
        assert step.reagents == [SmilesStr("O=S(=O)(Cl)Cl")]
        assert step.solvents == [SmilesStr("ClCCl")]
        assert step.metadata["patent_id"] == "US1234567"

    def test_single_reactant(self):
        """Test ReactionStep with single reactant."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant])
        assert len(step.reactants) == 1
        assert step.reactants[0] == reactant

    def test_empty_reactants_list(self):
        """Test that empty reactants list is allowed (edge case)."""
        step = ReactionStep(reactants=[])
        assert len(step.reactants) == 0

    def test_multiple_reagents_and_solvents(self):
        """Test ReactionStep with multiple reagents and solvents."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(
            reactants=[reactant],
            reagents=[SmilesStr("O"), SmilesStr("ClS(=O)(=O)Cl")],
            solvents=[SmilesStr("ClCCl"), SmilesStr("c1ccccc1")],  # DCM and benzene
        )
        assert len(step.reagents) == 2
        assert len(step.solvents) == 2
