import pytest

from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.exceptions import AdapterLogicError
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
PARACETAMOL_SMILES = "CC(=O)Nc1ccc(O)cc1"
DARIDOREXANT_SMILES = "COc1ccc(-n2nccn2)c(C(=O)N2CCC[C@@]2(C)c2nc3c(C)c(Cl)ccc3[nH]2)c1"


# ============================================================================
# Unit Tests: Test common failure modes
# ============================================================================


class TestRetroStarAdapterUnit(BaseAdapterTest):
    """Unit tests for RetroStarAdapter - tests common adapter failure modes."""

    @pytest.fixture
    def adapter_instance(self):
        return RetroStarAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        return {"succ": True, "routes": "CCO>0.9>CC=O.[H][H]"}

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return {"succ": False, "routes": ""}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # Adapter logic, not pydantic, will fail on a non-string route
        return {"succ": True, "routes": 123}

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")

    def test_parser_raises_on_invalid_step_format(self, adapter_instance):
        """The private parser method should raise an error for malformed steps."""
        bad_route_str = "CCO>CC=O"  # Missing the score part
        with pytest.raises(AdapterLogicError, match="Invalid format near"):
            adapter_instance._parse_route_string(bad_route_str)


# ============================================================================
# Contract Tests: Verify all routes meet schema requirements
# ============================================================================


@pytest.mark.contract
class TestRetroStarAdapterContract:
    """Contract tests - verify all routes meet schema requirements."""

    adapter = RetroStarAdapter()

    @pytest.fixture(scope="class")
    def aspirin_routes(self, raw_retrostar_data):
        """Run adapter once for aspirin and reuse results."""
        raw_data = raw_retrostar_data["aspirin"]
        target_input = TargetInput(id="aspirin", smiles=ASPIRIN_SMILES)
        return list(self.adapter.cast(raw_data, target_input))

    @pytest.fixture(scope="class")
    def paracetamol_routes(self, raw_retrostar_data):
        """Run adapter once for paracetamol and reuse results."""
        raw_data = raw_retrostar_data["paracetamol"]
        target_input = TargetInput(id="paracetamol", smiles=PARACETAMOL_SMILES)
        return list(self.adapter.cast(raw_data, target_input))

    @pytest.fixture(scope="class")
    def daridorexant_routes(self, raw_retrostar_data):
        """Run adapter once for daridorexant and reuse results."""
        raw_data = raw_retrostar_data["daridorexant"]
        target_input = TargetInput(id="daridorexant", smiles=DARIDOREXANT_SMILES)
        return list(self.adapter.cast(raw_data, target_input))

    def test_route_has_required_fields(self, aspirin_routes):
        """All routes must have required Route fields populated."""
        for route in aspirin_routes:
            assert route.target is not None
            assert route.rank > 0
            assert route.metadata is not None

    def test_molecule_has_required_fields(self, aspirin_routes):
        """All Molecule objects must have required fields populated."""
        for route in aspirin_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                assert mol.smiles is not None
                assert mol.inchikey is not None
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)

    def test_reaction_step_has_required_fields(self, aspirin_routes):
        """All ReactionStep objects must have required fields populated."""
        for route in aspirin_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                if mol.synthesis_step:
                    step = mol.synthesis_step
                    assert step.reactants is not None
                    assert len(step.reactants) > 0
                    molecules.extend(step.reactants)

    def test_leaf_molecules_have_no_synthesis_step(self, aspirin_routes):
        """Leaf molecules should not have a synthesis_step."""
        for route in aspirin_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                if mol.is_leaf:
                    assert mol.synthesis_step is None
                else:
                    assert mol.synthesis_step is not None
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)

    def test_route_metadata_contains_route_cost(self, aspirin_routes):
        """Route metadata should contain route_cost field."""
        for route in aspirin_routes:
            assert "route_cost" in route.metadata

    def test_purchasable_molecule_is_leaf(self, paracetamol_routes):
        """Purchasable molecules should be marked as leaf with no synthesis step."""
        for route in paracetamol_routes:
            assert route.target.is_leaf
            assert route.target.synthesis_step is None

    def test_multi_step_route_has_nested_structure(self, daridorexant_routes):
        """Multi-step routes should have nested ReactionStep structures."""
        for route in daridorexant_routes:
            # At least one intermediate (non-leaf) molecule should exist
            molecules = [route.target]
            has_intermediate = False
            while molecules:
                mol = molecules.pop()
                if not mol.is_leaf and mol.synthesis_step:
                    for reactant in mol.synthesis_step.reactants:
                        if not reactant.is_leaf:
                            has_intermediate = True
                            break
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)
            assert has_intermediate, "Multi-step route should have intermediate molecules"


# ============================================================================
# Regression Tests: Verify specific routes match expected values
# ============================================================================


@pytest.mark.regression
class TestRetroStarAdapterRegression:
    """Regression tests - verify specific routes match exact expected values."""

    adapter = RetroStarAdapter()

    def test_aspirin_single_step_route(self, raw_retrostar_data):
        """Verify exact structure of aspirin single-step route."""
        raw_data = raw_retrostar_data["aspirin"]
        target_input = TargetInput(id="aspirin", smiles=ASPIRIN_SMILES)

        routes = list(self.adapter.cast(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Target molecule
        assert target.smiles == ASPIRIN_SMILES
        assert not target.is_leaf
        assert target.synthesis_step is not None
        assert route.rank == 1

        # Route metadata
        assert route.metadata["route_cost"] == pytest.approx(0.5438278376934434)

        # Synthesis step
        synthesis_step = target.synthesis_step
        assert len(synthesis_step.reactants) == 2
        reactant_smiles = {r.smiles for r in synthesis_step.reactants}
        assert reactant_smiles == {"CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"}
        assert all(r.is_leaf for r in synthesis_step.reactants)

    def test_paracetamol_purchasable_route(self, raw_retrostar_data):
        """Verify exact structure of paracetamol purchasable route."""
        raw_data = raw_retrostar_data["paracetamol"]
        target_input = TargetInput(id="paracetamol", smiles=PARACETAMOL_SMILES)

        routes = list(self.adapter.cast(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Target molecule
        assert target.smiles == PARACETAMOL_SMILES
        assert target.is_leaf
        assert target.synthesis_step is None
        assert route.rank == 1

        # Route metadata
        assert route.metadata["route_cost"] == 0

    def test_daridorexant_multi_step_route(self, raw_retrostar_data):
        """Verify exact structure of daridorexant multi-step route."""
        raw_data = raw_retrostar_data["daridorexant"]
        target_input = TargetInput(id="daridorexant", smiles=DARIDOREXANT_SMILES)

        routes = list(self.adapter.cast(raw_data, target_input))
        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Level 1: daridorexant -> two precursors
        assert target.smiles == DARIDOREXANT_SMILES
        assert not target.is_leaf
        assert target.synthesis_step is not None
        assert route.rank == 1

        # Route metadata
        assert route.metadata["route_cost"] == pytest.approx(8.35212518356242)

        synthesis_step1 = target.synthesis_step
        assert len(synthesis_step1.reactants) == 2

        # Find the two branches
        branch1_smiles = "COc1ccc(-n2nccn2)c(C(=O)O)c1"
        branch2_smiles = "Cc1c(Cl)ccc2[nH]c([C@]3(C)CCCN3)nc12"

        branch1_mol = next(r for r in synthesis_step1.reactants if r.smiles == branch1_smiles)
        branch2_mol = next(r for r in synthesis_step1.reactants if r.smiles == branch2_smiles)

        # Level 2, branch 1: check that it decomposes further
        assert not branch1_mol.is_leaf
        assert branch1_mol.synthesis_step is not None
        synthesis_step2 = branch1_mol.synthesis_step
        assert len(synthesis_step2.reactants) == 2
        reactant_smiles_2 = {r.smiles for r in synthesis_step2.reactants}
        assert reactant_smiles_2 == {"COc1ccc(I)c(C(=O)O)c1", "c1cn[nH]n1"}
        assert all(r.is_leaf for r in synthesis_step2.reactants)

        # Level 2, branch 2: check that it also decomposes
        assert not branch2_mol.is_leaf
        assert branch2_mol.synthesis_step is not None
        synthesis_step3 = branch2_mol.synthesis_step
        assert len(synthesis_step3.reactants) == 2
        reactant_smiles_3 = {r.smiles for r in synthesis_step3.reactants}
        assert reactant_smiles_3 == {"C[C@@]1(C(=O)O)CCCN1", "Cc1c(Cl)ccc(N)c1N"}
        assert all(r.is_leaf for r in synthesis_step3.reactants)
