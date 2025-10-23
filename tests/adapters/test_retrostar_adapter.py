import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.retrostar_adapter import RetroStarAdapter
from ursa.domain.schemas import TargetInfo
from ursa.exceptions import AdapterLogicError

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
        # adapter logic, not pydantic, will fail on a non-string route
        return {"succ": True, "routes": 123}

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")

    def test_parser_raises_on_invalid_step_format(self, adapter_instance):
        """the private parser method should raise an error for malformed steps."""
        bad_route_str = "CCO>CC=O"  # missing the score part
        with pytest.raises(AdapterLogicError, match="invalid format near"):
            adapter_instance._parse_route_string(bad_route_str)


@pytest.mark.integration
class TestRetroStarAdapterIntegration:
    adapter = RetroStarAdapter()

    def test_adapt_single_step_route(self, raw_retrostar_data):
        """tests a successful, single-step route (aspirin)."""
        raw_data = raw_retrostar_data["aspirin"]
        target_info = TargetInfo(id="aspirin", smiles=ASPIRIN_SMILES)

        trees = list(self.adapter.adapt(raw_data, target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert tree.target.id == "aspirin"
        assert root.smiles == ASPIRIN_SMILES
        assert not root.is_starting_material
        assert len(root.reactions) == 1

        reaction = root.reactions[0]
        assert len(reaction.reactants) == 2
        reactant_smiles = {r.smiles for r in reaction.reactants}
        assert reactant_smiles == {"CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"}
        assert all(r.is_starting_material for r in reaction.reactants)

    def test_adapt_purchasable_molecule(self, raw_retrostar_data):
        """tests a target that is purchasable (paracetamol), resulting in a 0-step route."""
        raw_data = raw_retrostar_data["paracetamol"]
        target_info = TargetInfo(id="paracetamol", smiles=PARACETAMOL_SMILES)

        trees = list(self.adapter.adapt(raw_data, target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert root.smiles == PARACETAMOL_SMILES
        assert root.is_starting_material
        assert not root.reactions

    def test_adapt_multi_step_route(self, raw_retrostar_data):
        """tests a complex, multi-step route from a |-delimited string (daridorexant)."""
        raw_data = raw_retrostar_data["daridorexant"]
        target_info = TargetInfo(id="daridorexant", smiles=DARIDOREXANT_SMILES)

        trees = list(self.adapter.adapt(raw_data, target_info))
        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        # level 1: daridorexant -> two precursors
        assert root.smiles == DARIDOREXANT_SMILES
        assert not root.is_starting_material
        reaction1 = root.reactions[0]
        assert len(reaction1.reactants) == 2

        # find the two branches
        branch1_smiles = "COc1ccc(-n2nccn2)c(C(=O)O)c1"
        branch2_smiles = "Cc1c(Cl)ccc2[nH]c([C@]3(C)CCCN3)nc12"

        branch1_node = next(r for r in reaction1.reactants if r.smiles == branch1_smiles)
        branch2_node = next(r for r in reaction1.reactants if r.smiles == branch2_smiles)

        # level 2, branch 1: check that it decomposes further
        assert not branch1_node.is_starting_material
        reaction2 = branch1_node.reactions[0]
        assert len(reaction2.reactants) == 2
        reactant_smiles_2 = {r.smiles for r in reaction2.reactants}
        assert reactant_smiles_2 == {"COc1ccc(I)c(C(=O)O)c1", "c1cn[nH]n1"}
        assert all(r.is_starting_material for r in reaction2.reactants)

        # level 2, branch 2: check that it also decomposes
        assert not branch2_node.is_starting_material
        reaction3 = branch2_node.reactions[0]
        assert len(reaction3.reactants) == 2
        reactant_smiles_3 = {r.smiles for r in reaction3.reactants}
        assert reactant_smiles_3 == {"C[C@@]1(C(=O)O)CCCN1", "Cc1c(Cl)ccc(N)c1N"}
        assert all(r.is_starting_material for r in reaction3.reactants)
