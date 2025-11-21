"""Unit tests for Route.get_reaction_signatures() method."""

from typing import Any

import pytest

from retrocast.models.chem import Molecule, ReactionSignature, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr

# ==============================================================================
# Route.get_reaction_signatures Tests
# ==============================================================================


@pytest.mark.unit
class TestRouteGetReactionSignatures:
    """Tests for Route.get_reaction_signatures() method."""

    def test_leaf_route_returns_empty_set(self):
        """Test that a route with no reactions returns empty set."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1)

        signatures = route.get_reaction_signatures()
        assert signatures == set()
        assert isinstance(signatures, set)

    def test_single_step_route(self):
        """Test a route with single reaction step."""
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

        signatures = route.get_reaction_signatures()

        assert len(signatures) == 1
        sig = next(iter(signatures))
        assert isinstance(sig, tuple)
        assert len(sig) == 2
        reactant_keys, product_key = sig
        assert isinstance(reactant_keys, frozenset)
        assert product_key == "XEKOWRVHYACXOJ-UHFFFAOYSA-N"
        assert reactant_keys == frozenset(["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "QTBSBXVTEAMEQO-UHFFFAOYSA-N"])

    def test_multi_step_linear_route(self):
        """Test a linear route with multiple steps."""
        # Build from bottom up: leaf -> intermediate -> target
        leaf = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"))

        intermediate = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf]),
        )

        target = Molecule(
            smiles=SmilesStr("COC"),
            inchikey=InchiKeyStr("FAKE-KEY-1"),
            synthesis_step=ReactionStep(reactants=[intermediate]),
        )

        route = Route(target=target, rank=1)
        signatures = route.get_reaction_signatures()

        assert len(signatures) == 2

        # Check both reactions are present
        expected_sig1: ReactionSignature = (frozenset(["VNWKTOKETHGBQD-UHFFFAOYSA-N"]), "OKKJLVBELUTLKV-UHFFFAOYSA-N")
        expected_sig2: ReactionSignature = (frozenset(["OKKJLVBELUTLKV-UHFFFAOYSA-N"]), "FAKE-KEY-1")

        assert expected_sig1 in signatures
        assert expected_sig2 in signatures

    def test_branched_route(self):
        """Test a branched route with multiple parallel reactions."""
        # Left branch
        leaf1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"))
        intermediate_left = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("OKKJLVBELUTLKV-UHFFFAOYSA-N"),
            synthesis_step=ReactionStep(reactants=[leaf1]),
        )

        # Right branch (just a leaf)
        leaf2 = Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("QGZKDVFQNNGYKY-UHFFFAOYSA-N"))

        # Combine branches
        target = Molecule(
            smiles=SmilesStr("CON"),
            inchikey=InchiKeyStr("FAKE-KEY-3"),
            synthesis_step=ReactionStep(reactants=[intermediate_left, leaf2]),
        )

        route = Route(target=target, rank=1)
        signatures = route.get_reaction_signatures()

        # Should have 2 reactions: leaf1->intermediate_left, (intermediate_left + leaf2)->target
        assert len(signatures) == 2

        expected_sig1: ReactionSignature = (frozenset(["VNWKTOKETHGBQD-UHFFFAOYSA-N"]), "OKKJLVBELUTLKV-UHFFFAOYSA-N")
        expected_sig2: ReactionSignature = (
            frozenset(["OKKJLVBELUTLKV-UHFFFAOYSA-N", "QGZKDVFQNNGYKY-UHFFFAOYSA-N"]),
            "FAKE-KEY-3",
        )

        assert expected_sig1 in signatures
        assert expected_sig2 in signatures

    def test_reaction_signature_is_hashable(self):
        """Test that reaction signatures can be added to sets and used as dict keys."""
        sig: ReactionSignature = (frozenset(["KEY1", "KEY2"]), "KEY3")

        # Can be added to set
        sig_set = {sig}
        assert sig in sig_set

        # Can be used as dict key
        sig_dict = {sig: "test_value"}
        assert sig_dict[sig] == "test_value"

    def test_duplicate_reaction_in_route(self):
        """Test that same reaction appearing multiple times is deduplicated."""
        # This is an edge case where the same molecule (by InchiKey) is used in multiple places
        # but each has different synthesis paths - the reactions should be collected

        # Create a diamond-shaped dependency
        leaf1 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("LEAF1-KEY"))
        leaf2 = Molecule(smiles=SmilesStr("O"), inchikey=InchiKeyStr("LEAF2-KEY"))

        # Both branches produce intermediates
        intermediate1 = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("INTERMEDIATE1-KEY"),
            synthesis_step=ReactionStep(reactants=[leaf1]),
        )
        intermediate2 = Molecule(
            smiles=SmilesStr("OC"),
            inchikey=InchiKeyStr("INTERMEDIATE2-KEY"),
            synthesis_step=ReactionStep(reactants=[leaf2]),
        )

        # Final product combines both intermediates
        target = Molecule(
            smiles=SmilesStr("COC"),
            inchikey=InchiKeyStr("TARGET-KEY"),
            synthesis_step=ReactionStep(reactants=[intermediate1, intermediate2]),
        )

        route = Route(target=target, rank=1)
        signatures = route.get_reaction_signatures()

        # Should have 3 unique reactions
        assert len(signatures) == 3

    def test_comparing_routes_for_overlap(self):
        """Test the main use case: finding if routes share any reactions."""
        # Route 1: A -> B -> C
        leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
        intermediate_b = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("KEY-B"),
            synthesis_step=ReactionStep(reactants=[leaf_a]),
        )
        target_c = Molecule(
            smiles=SmilesStr("COC"),
            inchikey=InchiKeyStr("KEY-C"),
            synthesis_step=ReactionStep(reactants=[intermediate_b]),
        )
        route1 = Route(target=target_c, rank=1)

        # Route 2: A -> B -> D (shares first reaction with route1)
        leaf_a2 = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
        intermediate_b2 = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("KEY-B"),
            synthesis_step=ReactionStep(reactants=[leaf_a2]),
        )
        target_d = Molecule(
            smiles=SmilesStr("COCC"),
            inchikey=InchiKeyStr("KEY-D"),
            synthesis_step=ReactionStep(reactants=[intermediate_b2]),
        )
        route2 = Route(target=target_d, rank=1)

        # Route 3: X -> Y -> Z (no overlap with route1 or route2)
        leaf_x = Molecule(smiles=SmilesStr("N"), inchikey=InchiKeyStr("KEY-X"))
        intermediate_y = Molecule(
            smiles=SmilesStr("NC"),
            inchikey=InchiKeyStr("KEY-Y"),
            synthesis_step=ReactionStep(reactants=[leaf_x]),
        )
        target_z = Molecule(
            smiles=SmilesStr("NCC"),
            inchikey=InchiKeyStr("KEY-Z"),
            synthesis_step=ReactionStep(reactants=[intermediate_y]),
        )
        route3 = Route(target=target_z, rank=1)

        sigs1 = route1.get_reaction_signatures()
        sigs2 = route2.get_reaction_signatures()
        sigs3 = route3.get_reaction_signatures()

        # Routes 1 and 2 share the A->B reaction
        overlap_1_2 = sigs1 & sigs2
        assert len(overlap_1_2) == 1
        shared_reaction = next(iter(overlap_1_2))
        assert shared_reaction == (frozenset(["KEY-A"]), "KEY-B")

        # Routes 1 and 3 have no overlap
        overlap_1_3 = sigs1 & sigs3
        assert len(overlap_1_3) == 0

        # Routes 2 and 3 have no overlap
        overlap_2_3 = sigs2 & sigs3
        assert len(overlap_2_3) == 0

    def test_reaction_signature_type_alias(self):
        """Test that ReactionSignature type alias works correctly."""
        # Create a valid ReactionSignature
        sig: ReactionSignature = (frozenset(["KEY1", "KEY2"]), "KEY3")

        # Verify structure
        reactants, product = sig
        assert isinstance(reactants, frozenset)
        assert isinstance(product, str)

    def test_deep_route_with_pharma_data(self, pharma_routes_data: dict[str, Any]):
        """Test get_reaction_signatures with real pharma route data."""
        from tests.models.test_pharma_routes import TestPharmaRoutesContract

        helper = TestPharmaRoutesContract()
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        target_molecule = helper._build_molecule_tree(vonoprazan_data)
        route = Route(target=target_molecule, rank=1)

        signatures = route.get_reaction_signatures()

        # Vonoprazan has depth 2, so should have at least 2 reactions
        assert len(signatures) >= 2

        # All signatures should be valid tuples
        for sig in signatures:
            assert isinstance(sig, tuple)
            assert len(sig) == 2
            reactants, product = sig
            assert isinstance(reactants, frozenset)
            assert isinstance(product, str)
            assert len(reactants) > 0  # Must have at least one reactant

    def test_single_reactant_reaction(self):
        """Test a reaction with a single reactant."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant])
        target = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)

        signatures = route.get_reaction_signatures()

        assert len(signatures) == 1
        sig = next(iter(signatures))
        reactant_keys, product_key = sig
        assert reactant_keys == frozenset(["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"])
        assert product_key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"
