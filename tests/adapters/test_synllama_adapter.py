import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.synllama_adapter import SynLlaMaAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo
from ursa.exceptions import AdapterLogicError


class TestSynLlamaAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return SynLlaMaAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a single-step route for acetone from two precursors
        return [{"synthesis_string": "CC(=O)O;C;R1;CC(=O)C"}]

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
        return TargetInfo(id="acetone", smiles="CC(C)=O")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="acetone", smiles="CCC")

    def test_adapt_parses_multi_step_route(self, adapter_instance):
        """
        Tests that the adapter correctly parses a multi-step synthesis string
        and builds a tree with the correct depth and structure.

        Route: C -> CC -> CCC
        """
        multi_step_string = "C;R1;CC;C;R2;CCC"
        raw_data = [{"synthesis_string": multi_step_string}]
        target_info = TargetInfo(id="multi-step-test", smiles=canonicalize_smiles("CCC"))

        trees = list(adapter_instance.adapt(raw_data, target_info))

        assert len(trees) == 1
        root = trees[0].retrosynthetic_tree

        # Level 1: CCC -> CC + C
        assert root.smiles == "CCC"
        assert not root.is_starting_material
        reaction1 = root.reactions[0]
        assert {r.smiles for r in reaction1.reactants} == {"CC", "C"}

        # Level 2: find the intermediate (CC) and check its decomposition
        intermediate_node = next(r for r in reaction1.reactants if r.smiles == "CC")
        leaf_node = next(r for r in reaction1.reactants if r.smiles == "C")

        assert not intermediate_node.is_starting_material
        assert leaf_node.is_starting_material  # C is a starting material in this route

        reaction2 = intermediate_node.reactions[0]
        assert {r.smiles for r in reaction2.reactants} == {"C"}
        assert all(r.is_starting_material for r in reaction2.reactants)

    @pytest.mark.parametrize(
        "bad_string, error_match",
        [
            ("C;R1", "malformed route: template 'R1' has no product"),
            ("R1;C", "no reactants found for product 'C'"),
            ("", "synthesis string is empty."),
        ],
    )
    def test_parser_raises_on_invalid_string_format(self, adapter_instance, bad_string, error_match):
        """tests that the private parser method raises specific logic errors."""
        with pytest.raises(AdapterLogicError, match=error_match):
            adapter_instance._parse_synthesis_string(bad_string)


@pytest.mark.integration
class TestSynLlamaAdapterIntegration:
    adapter = SynLlaMaAdapter()

    @pytest.mark.parametrize(
        "target_id",
        ["Conivaptan hydrochloride", "AGN-190205", "USPTO-165/190"],
    )
    def test_adapt_synllama_routes(self, raw_synllama_data, target_id):
        """
        tests that the adapter correctly parses a valid, pre-processed synllama route.
        """
        raw_routes = raw_synllama_data[target_id]

        # derive the canonical target smiles from the raw data itself
        synthesis_str = raw_routes[0]["synthesis_string"]
        cleaned_parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
        product_smi_raw = cleaned_parts[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_info = TargetInfo(id=target_id, smiles=target_smi_canon)
        trees = list(self.adapter.adapt(raw_routes, target_info))

        assert len(trees) >= 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        # check root node properties
        assert tree.target.id == target_id
        assert root.smiles == target_smi_canon
        assert not root.is_starting_material
        assert len(root.reactions) == 1

        # check that the tree has some structure if it's not a 0-step route
        # (i.e., the parser found reactions)
        if not root.is_starting_material:
            assert len(root.reactions) > 0
            assert len(root.reactions[0].reactants) > 0
        else:
            assert len(root.reactions) == 0
