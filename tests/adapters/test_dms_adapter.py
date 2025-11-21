import logging

import pytest

from retrocast.adapters.dms_adapter import DMSAdapter, DMSTree
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


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
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        # correct id, wrong smiles
        return TargetInput(id="ethanol", smiles="CCC")

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
        target_input = TargetInput(id="ibuprofen_cycle_test", smiles=canonicalize_smiles(target_smiles))
        routes = list(adapter_instance.cast(cyclic_route_data, target_input))
        assert len(routes) == 0
        assert "cycle detected" in caplog.text


@pytest.mark.integration
class TestDMSAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> DMSAdapter:
        return DMSAdapter()

    @pytest.fixture(scope="class")
    def routes(self, adapter, raw_dms_data):
        """Shared fixture to avoid re-running adaptation for every test."""
        # Use ibuprofen data which has multiple routes
        raw_target_data = raw_dms_data["ibuprofen"]
        target_smiles = canonicalize_smiles(raw_target_data[0]["smiles"])
        target_input = TargetInput(id="ibuprofen", smiles=target_smiles)
        return list(adapter.cast(raw_target_data, target_input))

    def test_produces_multiple_routes(self, routes):
        """Verify the adapter produces multiple routes for ibuprofen."""
        assert len(routes) > 1

    def test_all_routes_have_ranks(self, routes):
        """Verify all routes are properly ranked and unique."""
        ranks = [route.rank for route in routes]
        # Ranks should be unique positive integers (but not necessarily consecutive due to filtering)
        assert all(isinstance(r, int) and r > 0 for r in ranks)
        assert len(ranks) == len(set(ranks))  # All ranks are unique

    def test_all_routes_have_inchikeys(self, routes):
        """Verify all target molecules have InChIKeys."""
        for route in routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_molecules_have_inchikeys(self, routes):
        """Verify all molecules in all routes have InChIKeys."""

        def check_molecule(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step is not None:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_route_depth_calculation_works(self, routes):
        """Verify the depth property is computed correctly."""
        for route in routes:
            # Just verify it doesn't crash and returns a non-negative integer
            depth = route.length
            assert isinstance(depth, int)
            assert depth >= 0


@pytest.mark.integration
class TestDMSAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> DMSAdapter:
        return DMSAdapter()

    def test_adapt_one_step_route(self, adapter, raw_dms_data):
        """Tests a simple, one-step route (aspirin)."""
        raw_route_data = raw_dms_data["aspirin"][0]
        target_smiles = canonicalize_smiles(raw_route_data["smiles"])
        target_input = TargetInput(id="aspirin", smiles=target_smiles)

        routes = list(adapter.cast([raw_route_data], target_input))

        assert len(routes) == 1
        route = routes[0]
        assert route.rank == 1

        target = route.target
        assert target.smiles == target_smiles
        assert not target.is_leaf
        assert target.synthesis_step is not None

        reaction = target.synthesis_step
        assert len(reaction.reactants) == 2

        # Derive expectations from the raw data
        expected_reactants_raw = [child["smiles"] for child in raw_route_data["children"]]
        expected_reactants_canon = {canonicalize_smiles(s) for s in expected_reactants_raw}
        actual_reactants_canon = {r.smiles for r in reaction.reactants}

        assert actual_reactants_canon == expected_reactants_canon
        assert all(r.is_leaf for r in reaction.reactants)

    def test_adapt_multi_step_route(self, adapter, raw_dms_data):
        """Tests a multi-step, linear route (paracetamol)."""
        raw_route_data = raw_dms_data["paracetamol"][0]
        target_smiles = canonicalize_smiles(raw_route_data["smiles"])
        target_input = TargetInput(id="paracetamol", smiles=target_smiles)

        routes = list(adapter.cast([raw_route_data], target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target
        assert target.smiles == target_smiles

        # --- level 1 ---
        reaction1 = target.synthesis_step
        assert reaction1 is not None
        assert len(reaction1.reactants) == 2

        # Find the intermediate programmatically from both raw and parsed data
        intermediate_raw = next(child for child in raw_route_data["children"] if child.get("children"))
        intermediate_canon_smiles = canonicalize_smiles(intermediate_raw["smiles"])
        intermediate_node = next(r for r in reaction1.reactants if not r.is_leaf)

        assert intermediate_node.smiles == intermediate_canon_smiles

        # --- level 2 ---
        reaction2 = intermediate_node.synthesis_step
        assert reaction2 is not None
        assert len(reaction2.reactants) == 1

        # Derive expectations for L2 from the raw intermediate
        expected_l2_reactant_raw = intermediate_raw["children"][0]["smiles"]
        expected_l2_reactant_canon = canonicalize_smiles(expected_l2_reactant_raw)
        actual_l2_reactant_canon = reaction2.reactants[0].smiles

        assert actual_l2_reactant_canon == expected_l2_reactant_canon
        assert reaction2.reactants[0].is_leaf

    def test_calculate_route_length(self, adapter, raw_dms_data):
        """Tests the static route length calculation for various route depths."""
        # case 0: a molecule with no children (starting material)
        dms_tree_0 = DMSTree(smiles="CCO", children=[])
        assert adapter.calculate_route_length(dms_tree_0) == 0

        # case 1: a one-step route (aspirin)
        route_len_1_raw = raw_dms_data["aspirin"][0]
        dms_tree_1 = DMSTree.model_validate(route_len_1_raw)
        assert adapter.calculate_route_length(dms_tree_1) == 1

        # case 2: a two-step route (paracetamol)
        route_len_2_raw = raw_dms_data["paracetamol"][0]
        dms_tree_2 = DMSTree.model_validate(route_len_2_raw)
        assert adapter.calculate_route_length(dms_tree_2) == 2

    def test_route_depth_matches_calculate_route_length(self, adapter, raw_dms_data):
        """Verify that the route.length property matches calculate_route_length."""
        for target_name in ["aspirin", "paracetamol"]:
            raw_route_data = raw_dms_data[target_name][0]
            target_smiles = canonicalize_smiles(raw_route_data["smiles"])
            target_input = TargetInput(id=target_name, smiles=target_smiles)

            routes = list(adapter.cast([raw_route_data], target_input))
            route = routes[0]

            dms_tree = DMSTree.model_validate(raw_route_data)
            expected_depth = adapter.calculate_route_length(dms_tree)

            assert route.length == expected_depth
