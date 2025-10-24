import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.synllama_adapter import SynLlaMaAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo


class TestSynLlamaAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return SynLlaMaAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # the pre-processed format is a list of route objects.
        return [{"synthesis_string": "CC=O;[H][H];SomeTemplateID;CCO"}]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # an empty list, from a target with no routes found.
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # wrong key in the dict, will fail pydantic validation.
        return [{"invalid_key": "..."}]

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestSynLlamaAdapterIntegration:
    adapter = SynLlaMaAdapter()

    @pytest.mark.parametrize(
        "target_id",
        ["Conivaptan hydrochloride", "AGN-190205"],
    )
    def test_adapt_synllama_routes(self, raw_synllama_data, target_id):
        """
        tests that the adapter correctly parses a valid, pre-processed synllama route.
        """
        raw_routes = raw_synllama_data[target_id]

        # derive the canonical target smiles from the raw data itself
        synthesis_str = raw_routes[0]["synthesis_string"]
        product_smi_raw = synthesis_str.split(";")[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_info = TargetInfo(id=target_id, smiles=target_smi_canon)
        trees = list(self.adapter.adapt(raw_routes, target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        # check root node properties
        assert tree.target.id == target_id
        assert root.smiles == target_smi_canon
        assert not root.is_starting_material
        assert len(root.reactions) == 1

        # check reaction properties
        reaction = root.reactions[0]
        reactants_raw = synthesis_str.split(";")[:-2]
        expected_reactants_canon = {canonicalize_smiles(s) for s in reactants_raw}
        actual_reactants_canon = {r.smiles for r in reaction.reactants}

        assert len(reaction.reactants) == len(expected_reactants_canon)
        assert actual_reactants_canon == expected_reactants_canon
        assert all(r.is_starting_material for r in reaction.reactants)
