import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.dms_adapter import DMSAdapter, DMSTree
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo


class TestDMSAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return DMSAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a list containing a single, one-step route
        return [
            {
                "smiles": "CCO",
                "children": [
                    {"smiles": "CC=O", "children": []},
                    {"smiles": "[H][H]", "children": []},
                ],
            }
        ]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # for dms, "unsuccessful" just means an empty list of routes
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # 'children' is a string, which fails pydantic validation
        return [{"smiles": "CCO", "children": "not a list"}]

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        # correct id, wrong smiles
        return TargetInfo(id="ethanol", smiles="CCC")

    def test_adapter_handles_cyclic_route_gracefully(self, adapter_instance, caplog):
        """proves the dms adapter's cycle detection correctly discards the invalid route."""
        target_smiles = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
        cyclic_route_data = [
            {
                "smiles": target_smiles,
                "children": [
                    {
                        "smiles": "CC(C)c1ccccc1",  # intermediate
                        "children": [{"smiles": target_smiles, "children": []}],  # <-- cycle
                    }
                ],
            }
        ]
        target_info = TargetInfo(id="ibuprofen_cycle_test", smiles=canonicalize_smiles(target_smiles))
        trees = list(adapter_instance.adapt(cyclic_route_data, target_info))
        assert len(trees) == 0
        assert "cycle detected" in caplog.text


@pytest.mark.integration
class TestDMSAdapterIntegration:
    adapter = DMSAdapter()

    def test_adapt_one_step_route(self, raw_dms_data):
        """tests a simple, one-step route (aspirin)."""
        raw_route_data = raw_dms_data["aspirin"][0]
        target_smiles = canonicalize_smiles(raw_route_data["smiles"])
        target_info = TargetInfo(id="aspirin", smiles=target_smiles)

        trees = list(self.adapter.adapt([raw_route_data], target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree
        reaction = root.reactions[0]

        # derive expectations from the raw data
        expected_reactants_raw = [child["smiles"] for child in raw_route_data["children"]]
        expected_reactants_canon = {canonicalize_smiles(s) for s in expected_reactants_raw}
        actual_reactants_canon = {r.smiles for r in reaction.reactants}

        assert root.smiles == target_smiles
        assert actual_reactants_canon == expected_reactants_canon
        assert all(r.is_starting_material for r in reaction.reactants)

    def test_adapt_multi_step_route(self, raw_dms_data):
        """tests a multi-step, linear route (paracetamol)."""
        raw_route_data = raw_dms_data["paracetamol"][0]
        target_smiles = canonicalize_smiles(raw_route_data["smiles"])
        target_info = TargetInfo(id="paracetamol", smiles=target_smiles)

        trees = list(self.adapter.adapt([raw_route_data], target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree
        assert root.smiles == target_smiles

        # --- level 1 ---
        reaction1 = root.reactions[0]
        assert len(reaction1.reactants) == 2

        # find the intermediate programmatically from both raw and parsed data
        intermediate_raw = next(child for child in raw_route_data["children"] if child.get("children"))
        intermediate_canon_smiles = canonicalize_smiles(intermediate_raw["smiles"])
        intermediate_node = next(r for r in reaction1.reactants if not r.is_starting_material)

        assert intermediate_node.smiles == intermediate_canon_smiles

        # --- level 2 ---
        reaction2 = intermediate_node.reactions[0]
        assert len(reaction2.reactants) == 1

        # derive expectations for L2 from the raw intermediate
        expected_l2_reactant_raw = intermediate_raw["children"][0]["smiles"]
        expected_l2_reactant_canon = canonicalize_smiles(expected_l2_reactant_raw)
        actual_l2_reactant_canon = reaction2.reactants[0].smiles

        assert actual_l2_reactant_canon == expected_l2_reactant_canon
        assert reaction2.reactants[0].is_starting_material

    def test_calculate_route_length(self, raw_dms_data):
        """tests the static route length calculation for various route depths."""
        # case 0: a molecule with no children (starting material)
        dms_tree_0 = DMSTree(smiles="CCO", children=[])
        assert self.adapter.calculate_route_length(dms_tree_0) == 0

        # case 1: a one-step route (aspirin)
        route_len_1_raw = raw_dms_data["aspirin"][0]
        dms_tree_1 = DMSTree.model_validate(route_len_1_raw)
        assert self.adapter.calculate_route_length(dms_tree_1) == 1

        # case 2: a two-step route (paracetamol)
        route_len_2_raw = raw_dms_data["paracetamol"][0]
        dms_tree_2 = DMSTree.model_validate(route_len_2_raw)
        assert self.adapter.calculate_route_length(dms_tree_2) == 2
