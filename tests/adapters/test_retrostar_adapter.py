import pytest

from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.exceptions import AdapterLogicError
from retrocast.schemas import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
PARACETAMOL_SMILES = "CC(=O)Nc1ccc(O)cc1"
DARIDOREXANT_SMILES = "COc1ccc(-n2nccn2)c(C(=O)N2CCC[C@@]2(C)c2nc3c(C)c(Cl)ccc3[nH]2)c1"


class TestRetroStarAdapterUnit(BaseAdapterTest):
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


@pytest.mark.integration
class TestRetroStarAdapterIntegration:
    adapter = RetroStarAdapter()

    def test_adapt_single_step_route(self, raw_retrostar_data):
        """Tests a successful, single-step route (aspirin)."""
        raw_data = raw_retrostar_data["aspirin"]
        target_input = TargetInput(id="aspirin", smiles=ASPIRIN_SMILES)

        routes = list(self.adapter.adapt(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        assert target.smiles == ASPIRIN_SMILES
        assert target.inchikey  # Ensure InChIKey is populated
        assert not target.is_leaf
        assert target.synthesis_step is not None
        assert route.rank == 1

        # Check that route_cost metadata is captured
        assert "route_cost" in route.metadata
        assert route.metadata["route_cost"] == pytest.approx(0.5438278376934434)

        synthesis_step = target.synthesis_step
        assert len(synthesis_step.reactants) == 2
        reactant_smiles = {r.smiles for r in synthesis_step.reactants}
        assert reactant_smiles == {"CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"}
        assert all(r.is_leaf for r in synthesis_step.reactants)

    def test_adapt_purchasable_molecule(self, raw_retrostar_data):
        """Tests a target that is purchasable (paracetamol), resulting in a 0-step route."""
        raw_data = raw_retrostar_data["paracetamol"]
        target_input = TargetInput(id="paracetamol", smiles=PARACETAMOL_SMILES)

        routes = list(self.adapter.adapt(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        assert target.smiles == PARACETAMOL_SMILES
        assert target.inchikey
        assert target.is_leaf
        assert target.synthesis_step is None
        assert route.rank == 1

        # Check that route_cost metadata is captured (0 for purchasable)
        assert "route_cost" in route.metadata
        assert route.metadata["route_cost"] == 0

    def test_adapt_multi_step_route(self, raw_retrostar_data):
        """Tests a complex, multi-step route from a |-delimited string (daridorexant)."""
        raw_data = raw_retrostar_data["daridorexant"]
        target_input = TargetInput(id="daridorexant", smiles=DARIDOREXANT_SMILES)

        routes = list(self.adapter.adapt(raw_data, target_input))
        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Level 1: daridorexant -> two precursors
        assert target.smiles == DARIDOREXANT_SMILES
        assert not target.is_leaf
        assert target.synthesis_step is not None
        assert route.rank == 1

        # Check that route_cost metadata is captured
        assert "route_cost" in route.metadata
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
