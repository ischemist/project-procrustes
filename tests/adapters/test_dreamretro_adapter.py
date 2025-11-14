import pytest

from retrocast.adapters.dreamretro_adapter import DreamRetroAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.schemas import TargetInput
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
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")

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

        target_input = TargetInput(id="Mirabegron", smiles=canonicalize_smiles(product_smi_raw))
        routes = list(self.adapter.adapt(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        root = route.target

        assert root.smiles == canonicalize_smiles(product_smi_raw)
        assert root.synthesis_step is not None
        reaction = root.synthesis_step

        reactant_smiles = {r.smiles for r in reaction.reactants}
        expected_reactants = {canonicalize_smiles(s) for s in reactants_smi_raw.split(".")}

        assert reactant_smiles == expected_reactants

        # verify metadata
        assert route.metadata["expand_model_call"] == raw_data["expand_model_call"]
        assert route.metadata["value_model_call"] == raw_data["value_model_call"]
        assert route.metadata["reaction_nodes_lens"] == raw_data["reaction_nodes_lens"]
        assert route.metadata["mol_nodes_lens"] == raw_data["mol_nodes_lens"]

    def test_adapt_multi_step_route(self, raw_dreamretro_data):
        """tests a multi-step route from a |-delimited string (anagliptin)."""
        raw_data = raw_dreamretro_data["Anagliptin"]
        raw_route_str = raw_data["routes"]
        root_smi_raw, _, _ = raw_route_str.split(">>")[0].split("|")[0].partition(">")
        target_input = TargetInput(id="Anagliptin", smiles=canonicalize_smiles(root_smi_raw))

        routes = list(self.adapter.adapt(raw_data, target_input))
        assert len(routes) == 1
        root = routes[0].target

        assert root.smiles == canonicalize_smiles(root_smi_raw)
        assert root.synthesis_step is not None
        reaction1 = root.synthesis_step
        assert len(reaction1.reactants) == 2

        intermediate_node = next(r for r in reaction1.reactants if not r.is_leaf)
        leaf_node = next(r for r in reaction1.reactants if r.is_leaf)
        assert leaf_node.smiles == "N#C[C@@H]1CCCN1C(=O)CCl"

        assert intermediate_node.synthesis_step is not None
        reaction2 = intermediate_node.synthesis_step
        assert len(reaction2.reactants) == 2
        assert all(r.is_leaf for r in reaction2.reactants)

        reactant_smiles_l2 = {r.smiles for r in reaction2.reactants}
        _, l2_reactants_raw = raw_route_str.split("|")[1].split(">>")
        expected_reactants_l2 = {canonicalize_smiles(s) for s in l2_reactants_raw.split(".")}
        assert reactant_smiles_l2 == expected_reactants_l2

        # verify metadata
        assert routes[0].metadata["expand_model_call"] == raw_data["expand_model_call"]
        assert routes[0].metadata["value_model_call"] == raw_data["value_model_call"]
        assert routes[0].metadata["reaction_nodes_lens"] == raw_data["reaction_nodes_lens"]
        assert routes[0].metadata["mol_nodes_lens"] == raw_data["mol_nodes_lens"]
