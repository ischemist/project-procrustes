"""Unit tests for Route class."""

from retrocast.schemas import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr

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
