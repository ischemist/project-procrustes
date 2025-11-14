"""Unit tests for retrocast.schemas module."""

from typing import Any

import pytest
from pydantic import ValidationError

from retrocast.domain.chem import canonicalize_smiles, get_inchi_key
from retrocast.schemas import Molecule, ReactionStep, Route, TargetInput
from retrocast.typing import InchiKeyStr, ReactionSmilesStr, SmilesStr

# ==============================================================================
# TargetInput Tests
# ==============================================================================


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


# ==============================================================================
# Route Tests
# ==============================================================================


class TestRoute:
    """Tests for the Route class."""

    def test_basic_instantiation(self):
        """Test creating a basic Route."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1)
        assert route.target == target
        assert route.rank == 1
        assert route.solvability == {}
        assert route.metadata == {}

    def test_depth_single_leaf(self):
        """Test depth calculation for single leaf molecule."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1)
        assert route.depth == 0

    def test_depth_single_step(self):
        """Test depth calculation for single synthesis step."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant1, reactant2])
        target = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)
        assert route.depth == 1

    def test_depth_multi_step_linear(self):
        """Test depth calculation for multi-step linear route."""
        # Build from bottom up
        leaf = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"))

        intermediate1 = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf]),
        )

        intermediate2 = Molecule(
            smiles=SmilesStr("COC"),
            inchikey=InchiKeyStr("FAKE-KEY-1"),
            synthesis_step=ReactionStep(reactants=[intermediate1]),
        )

        target = Molecule(
            smiles=SmilesStr("COCOC"),
            inchikey=InchiKeyStr("FAKE-KEY-2"),
            synthesis_step=ReactionStep(reactants=[intermediate2]),
        )

        route = Route(target=target, rank=1)
        assert route.depth == 3

    def test_depth_branched_route(self):
        """Test depth calculation for branched route (should return max depth)."""
        # Left branch: depth 2
        leaf1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"))
        intermediate_left = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf1]),
        )

        # Right branch: depth 1 (just a leaf)
        leaf2 = Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("QGZKDVFQNNGYKY-UHFFFAOYSA-N"))

        # Combine branches
        target = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("FAKE-KEY-3"),
            synthesis_step=ReactionStep(reactants=[intermediate_left, leaf2]),
        )

        route = Route(target=target, rank=1)
        # Max depth: 1 (to intermediate_left) + 1 (to leaf1) = 2 for left branch
        # Right branch is just 1 (to leaf2)
        # So max is 2
        assert route.depth == 2

    def test_leaves_property_single_leaf(self):
        """Test leaves property for single leaf molecule."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1)
        assert route.leaves == {target}
        assert len(route.leaves) == 1

    def test_leaves_property_simple_route(self):
        """Test leaves property for simple route."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant1, reactant2])
        target = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)
        assert route.leaves == {reactant1, reactant2}
        assert len(route.leaves) == 2

    def test_leaves_property_deduplication(self):
        """Test that leaves property deduplicates correctly."""
        # Same reactant used in different branches
        common_reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        unique_reactant = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )

        step = ReactionStep(reactants=[common_reactant, common_reactant, unique_reactant])
        target = Molecule(
            smiles=SmilesStr("FAKE-PRODUCT"),
            inchikey=InchiKeyStr("FAKE-KEY-4"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)
        assert len(route.leaves) == 2  # Deduplicated
        assert common_reactant in route.leaves
        assert unique_reactant in route.leaves

    def test_get_signature_deterministic(self):
        """Test that get_signature produces deterministic results."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant1, reactant2])
        target = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)

        sig1 = route.get_signature()
        sig2 = route.get_signature()
        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert len(sig1) == 64  # SHA256 hex digest length

    def test_get_signature_identical_routes(self):
        """Test that identical routes have same signature."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )

        # Create first route
        step1 = ReactionStep(reactants=[reactant1, reactant2])
        target1 = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step1,
        )
        route1 = Route(target=target1, rank=1)

        # Create identical second route (different objects, same structure)
        reactant1_copy = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2_copy = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step2 = ReactionStep(reactants=[reactant1_copy, reactant2_copy])
        target2 = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step2,
        )
        route2 = Route(target=target2, rank=2)

        assert route1.get_signature() == route2.get_signature()

    def test_get_signature_different_routes(self):
        """Test that different routes have different signatures."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )

        # First route
        step1 = ReactionStep(reactants=[reactant1, reactant2])
        target1 = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step1,
        )
        route1 = Route(target=target1, rank=1)

        # Different route (different reactants)
        reactant3 = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
        )
        step2 = ReactionStep(reactants=[reactant3])
        target2 = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=step2,
        )
        route2 = Route(target=target2, rank=1)

        assert route1.get_signature() != route2.get_signature()

    def test_get_signature_order_invariance(self):
        """Test that reactant order doesn't affect signature."""
        reactant1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2 = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )

        # First order
        step1 = ReactionStep(reactants=[reactant1, reactant2])
        target1 = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step1,
        )
        route1 = Route(target=target1, rank=1)

        # Reversed order
        reactant1_copy = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        reactant2_copy = Molecule(
            smiles=SmilesStr("CC(=O)O"),
            inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
        )
        step2 = ReactionStep(reactants=[reactant2_copy, reactant1_copy])  # Reversed
        target2 = Molecule(
            smiles=SmilesStr("CCOC(C)=O"),
            inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
            synthesis_step=step2,
        )
        route2 = Route(target=target2, rank=1)

        # Should be the same due to sorting in get_signature
        assert route1.get_signature() == route2.get_signature()

    def test_get_signature_with_repeated_molecule(self):
        """Test get_signature with the same molecule appearing multiple times (tests memoization)."""
        # Branch 1: intermediate is formed from leaf1
        leaf1 = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
        )
        branch1_intermediate = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf1]),
        )

        # Branch 2: same molecule (by InChIKey) formed from leaf2
        # This tests memoization in get_signature when the same InChIKey appears in different branches
        leaf2 = Molecule(
            smiles=SmilesStr("O"),
            inchikey=InchiKeyStr("XLYOFNOQVPJJNP-UHFFFAOYSA-M"),
        )
        branch2_intermediate = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf2]),
        )

        # Combine branches into final product
        target = Molecule(
            smiles=SmilesStr("CCOCCO"),
            inchikey=InchiKeyStr("MTHSVFCYNBDYFN-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[branch1_intermediate, branch2_intermediate]),
        )

        route = Route(target=target, rank=1)
        signature = route.get_signature()

        # Should produce a valid signature
        assert isinstance(signature, str)
        assert len(signature) == 64

    def test_solvability_field(self):
        """Test solvability field handling."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1, solvability={"emolecules": True, "mcule": False})
        assert route.solvability["emolecules"] is True
        assert route.solvability["mcule"] is False

    def test_metadata_handling(self):
        """Test route-level metadata."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        metadata = {"total_score": 0.95, "search_time": 42.5}
        route = Route(target=target, rank=1, metadata=metadata)
        assert route.metadata["total_score"] == 0.95
        assert route.metadata["search_time"] == 42.5


# ==============================================================================
# Contract/Regression Tests using Pharma Routes
# ==============================================================================


class TestPharmaRoutesContract:
    """Contract tests using real pharmaceutical route data."""

    def _build_molecule_tree(self, node_data: dict[str, Any]) -> Molecule:
        """Recursively build a Molecule tree from pharma routes JSON structure."""
        smiles = canonicalize_smiles(node_data["smiles"])
        inchikey = get_inchi_key(smiles)

        # Check if this node has children (i.e., it's not a leaf)
        if "children" in node_data and node_data["children"]:
            # Recursively build reactant molecules
            reactants = [self._build_molecule_tree(child) for child in node_data["children"]]
            synthesis_step = ReactionStep(reactants=reactants)
            return Molecule(smiles=smiles, inchikey=inchikey, synthesis_step=synthesis_step)
        else:
            # Leaf node
            return Molecule(smiles=smiles, inchikey=inchikey)

    def test_vonoprazan_route_structure(self, pharma_routes_data: dict[str, Any]):
        """Test building and analyzing vonoprazan-1 route."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        target_molecule = self._build_molecule_tree(vonoprazan_data)
        route = Route(target=target_molecule, rank=1)

        # Verify route properties
        assert route.rank == 1
        assert not route.target.is_leaf
        assert route.depth > 0

        # Expected leaves based on pharma_routes.json structure
        # Vonoprazan has 3 leaf nodes: "O=Cc1c[nH]c(-c2ccccc2F)c1", "O=S(=O)(Cl)c1cccnc1", "CN"
        assert len(route.leaves) == 3

        # Verify signature is deterministic
        sig1 = route.get_signature()
        sig2 = route.get_signature()
        assert sig1 == sig2

    def test_mitapivat_route_structure(self, pharma_routes_data: dict[str, Any]):
        """Test building and analyzing mitapivat-1 route (deeper tree)."""
        mitapivat_data = pharma_routes_data["mitapivat-1"]
        target_molecule = self._build_molecule_tree(mitapivat_data)
        route = Route(target=target_molecule, rank=1)

        # Verify route properties
        assert route.rank == 1
        assert not route.target.is_leaf
        assert route.depth >= 3  # Mitapivat has a deeper tree

        # Count leaves
        leaves = route.leaves
        assert len(leaves) > 0

        # All leaves should be leaf molecules
        for leaf in leaves:
            assert leaf.is_leaf

    def test_pharma_routes_signature_uniqueness(self, pharma_routes_data: dict[str, Any]):
        """Test that different pharma routes have different signatures."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        mitapivat_data = pharma_routes_data["mitapivat-1"]

        vonoprazan_target = self._build_molecule_tree(vonoprazan_data)
        mitapivat_target = self._build_molecule_tree(mitapivat_data)

        route1 = Route(target=vonoprazan_target, rank=1)
        route2 = Route(target=mitapivat_target, rank=1)

        # Different routes should have different signatures
        assert route1.get_signature() != route2.get_signature()

    def test_pharma_routes_roundtrip(self, pharma_routes_data: dict[str, Any]):
        """Test that pharma routes can be built, serialized, and reconstructed."""
        for route_id, route_data in pharma_routes_data.items():
            # Build route from JSON data
            target_molecule = self._build_molecule_tree(route_data)
            original_route = Route(target=target_molecule, rank=1)

            # Serialize to dict
            route_dict = original_route.model_dump(exclude={"leaves", "depth"})
            assert "target" in route_dict
            assert "rank" in route_dict
            assert route_dict["rank"] == 1

            # Reconstruct from dict (this is the "round trip")
            reconstructed_route = Route.model_validate(route_dict)

            # Verify reconstructed route matches original
            assert reconstructed_route.rank == original_route.rank
            assert reconstructed_route.target.smiles == original_route.target.smiles
            assert reconstructed_route.target.inchikey == original_route.target.inchikey

            # Verify signatures match (proves tree structure is preserved)
            assert reconstructed_route.get_signature() == original_route.get_signature(), (
                f"Route {route_id}: signatures don't match after roundtrip"
            )

            # Verify target SMILES matches original data (after canonicalization)
            expected_smiles = canonicalize_smiles(route_data["smiles"])
            assert reconstructed_route.target.smiles == expected_smiles

    def test_vonoprazan_depth_calculation(self, pharma_routes_data: dict[str, Any]):
        """Test specific depth calculation for vonoprazan route."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        target_molecule = self._build_molecule_tree(vonoprazan_data)
        route = Route(target=target_molecule, rank=1)

        # Based on the structure in pharma_routes.json:
        # vonoprazan-1 has 2 levels of reactions
        # Level 1: target -> [intermediate, "CN"]
        # Level 2: intermediate -> ["O=Cc1c[nH]c(-c2ccccc2F)c1", "O=S(=O)(Cl)c1cccnc1"]
        assert route.depth == 2

    def test_mitapivat_depth_calculation(self, pharma_routes_data: dict[str, Any]):
        """Test specific depth calculation for mitapivat route."""
        mitapivat_data = pharma_routes_data["mitapivat-1"]
        target_molecule = self._build_molecule_tree(mitapivat_data)
        route = Route(target=target_molecule, rank=1)

        # Mitapivat has a deeper tree structure
        # Should have depth >= 4 based on the nested structure
        assert route.depth >= 4
