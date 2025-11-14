import pytest

from retrocast.adapters.synplanner_adapter import SynPlannerAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import TargetInfo
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestSynPlannerAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return SynPlannerAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a minimal, valid bipartite graph route
        return [
            {
                "smiles": "CCO",
                "type": "mol",
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "...",
                        "children": [{"smiles": "CC=O", "type": "mol", "in_stock": True, "children": []}],
                    }
                ],
            }
        ]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # missing 'type' discriminator key
        return [{"smiles": "CCO", "children": []}]

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestSynPlannerAdapterIntegration:
    adapter = SynPlannerAdapter()

    def test_adapt_handles_target_with_no_routes(self, raw_synplanner_data):
        """adapter should yield nothing for a target with an empty list of routes."""
        raw_data = raw_synplanner_data["ibuprofen"]
        target_info = TargetInfo(id="ibuprofen", smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1")

        trees = list(self.adapter.adapt(raw_data, target_info))
        assert len(trees) == 0

    def test_adapt_parses_all_routes_for_target(self, raw_synplanner_data):
        """adapter should yield a tree for each route object in the input list."""
        raw_data = raw_synplanner_data["paracetamol"]
        # derive smiles from the first route to ensure a match
        target_smi_raw = raw_data[0]["smiles"]
        target_info = TargetInfo(id="paracetamol", smiles=canonicalize_smiles(target_smi_raw))

        trees = list(self.adapter.adapt(raw_data, target_info))
        assert len(trees) == len(raw_data)

    def test_adapt_parses_multi_step_route_correctly(self, raw_synplanner_data):
        """performs a deep check on a single, multi-step route's structure."""
        raw_route = raw_synplanner_data["paracetamol"][0]
        target_smi_raw = raw_route["smiles"]
        target_info = TargetInfo(id="paracetamol", smiles=canonicalize_smiles(target_smi_raw))

        # the adapter expects a list, so wrap the single route
        trees = list(self.adapter.adapt([raw_route], target_info))
        assert len(trees) == 1
        root = trees[0].retrosynthetic_tree

        # level 1: paracetamol
        assert root.smiles == canonicalize_smiles(target_smi_raw)
        reaction1 = root.reactions[0]
        assert len(reaction1.reactants) == 2

        # level 2: find the intermediate
        raw_intermediate_l2 = next(child for child in raw_route["children"][0]["children"] if not child["in_stock"])
        intermediate_l2_smiles = canonicalize_smiles(raw_intermediate_l2["smiles"])
        intermediate_l2 = next(r for r in reaction1.reactants if not r.is_starting_material)
        assert intermediate_l2.smiles == intermediate_l2_smiles
        reaction2 = intermediate_l2.reactions[0]
        assert len(reaction2.reactants) == 2

        # level 3: find the next intermediate
        raw_intermediate_l3 = next(
            child for child in raw_intermediate_l2["children"][0]["children"] if not child["in_stock"]
        )
        intermediate_l3 = next(r for r in reaction2.reactants if not r.is_starting_material)
        assert intermediate_l3.smiles == canonicalize_smiles(raw_intermediate_l3["smiles"])
        reaction3 = intermediate_l3.reactions[0]
        assert len(reaction3.reactants) == 1

        # level 4: the final leaf node
        leaf = reaction3.reactants[0]
        assert leaf.is_starting_material
        assert leaf.smiles == "CC(C)(C)Oc1ccccc1"
