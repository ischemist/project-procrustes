import pytest

from retrocast.adapters.dreamretro_adapter import DreamRetroAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.DEPRECATE_schemas import TargetInfo
from retrocast.exceptions import AdapterLogicError
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestDreamRetroAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return DreamRetroAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        return {"succ": True, "routes": "CCO>>CC=O.[H][H]"}

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return {"succ": False, "routes": "CCO>>CC=O.[H][H]"}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # the adapter logic, not pydantic, checks for dict and 'succ' key
        return "this is not a dict"

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")

    def test_parser_raises_on_invalid_step_format(self, adapter_instance):
        """the private parser method should raise an error for malformed steps."""
        bad_route_str = "CCO>CC=O.O"  # missing a ">"
        with pytest.raises(AdapterLogicError, match="invalid format near"):
            adapter_instance._parse_route_string(bad_route_str)


@pytest.mark.integration
class TestDreamRetroAdapterIntegration:
    adapter = DreamRetroAdapter()

    def test_adapt_single_step_route(self, raw_dreamretro_data):
        """tests a successful, single-step route (mirabegron)."""
        raw_data = raw_dreamretro_data["Mirabegron"]
        raw_route_str = raw_data["routes"]

        product_smi_raw, reactants_smi_raw = raw_route_str.split(">>")

        target_info = TargetInfo(id="Mirabegron", smiles=canonicalize_smiles(product_smi_raw))
        trees = list(self.adapter.adapt(raw_data, target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert root.smiles == canonicalize_smiles(product_smi_raw)
        reaction = root.reactions[0]

        reactant_smiles = {r.smiles for r in reaction.reactants}
        expected_reactants = {canonicalize_smiles(s) for s in reactants_smi_raw.split(".")}

        assert reactant_smiles == expected_reactants

    def test_adapt_multi_step_route(self, raw_dreamretro_data):
        """tests a multi-step route from a |-delimited string (anagliptin)."""
        raw_data = raw_dreamretro_data["Anagliptin"]
        raw_route_str = raw_data["routes"]
        root_smi_raw, _, _ = raw_route_str.split(">>")[0].split("|")[0].partition(">")
        target_info = TargetInfo(id="Anagliptin", smiles=canonicalize_smiles(root_smi_raw))

        trees = list(self.adapter.adapt(raw_data, target_info))
        assert len(trees) == 1
        root = trees[0].retrosynthetic_tree

        assert root.smiles == canonicalize_smiles(root_smi_raw)
        reaction1 = root.reactions[0]
        assert len(reaction1.reactants) == 2

        intermediate_node = next(r for r in reaction1.reactants if not r.is_starting_material)
        leaf_node = next(r for r in reaction1.reactants if r.is_starting_material)
        assert leaf_node.smiles == "N#C[C@@H]1CCCN1C(=O)CCl"

        reaction2 = intermediate_node.reactions[0]
        assert len(reaction2.reactants) == 2
        assert all(r.is_starting_material for r in reaction2.reactants)

        reactant_smiles_l2 = {r.smiles for r in reaction2.reactants}
        _, l2_reactants_raw = raw_route_str.split("|")[1].split(">>")
        expected_reactants_l2 = {canonicalize_smiles(s) for s in l2_reactants_raw.split(".")}
        assert reactant_smiles_l2 == expected_reactants_l2
