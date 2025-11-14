import pytest

from retrocast.adapters.multistepttl_adapter import TtlRetroAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import TargetInfo
from retrocast.utils.serializers import serialize_multistepttl_directory
from tests.adapters.test_base_adapter import BaseAdapterTest

IBUPROFEN_SMILES = canonicalize_smiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")


class TestTtlRetroAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return TtlRetroAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        return [
            {
                "reactions": [{"product": "CCO", "reactants": ["CC=O", "[H][H]"]}],
                "metadata": {"steps": 1},
            }
        ]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # an empty list of routes
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # 'reactions' key contains a string, not a list
        return [{"reactions": "not a list"}]

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestTtlRetroAdapterIntegration:
    adapter = TtlRetroAdapter()

    @pytest.fixture(scope="class")
    def serialized_ibuprofen_data(self, multistepttl_ibuprofen_dir) -> list[dict]:
        """serializes the ibuprofen pickle data once for all tests in this module."""
        data = serialize_multistepttl_directory(multistepttl_ibuprofen_dir)
        assert data is not None, "serialization failed for ibuprofen"
        return data

    @pytest.mark.filterwarnings("ignore:numpy.core.numeric is deprecated:DeprecationWarning")
    def test_adapt_parses_all_routes(self, serialized_ibuprofen_data):
        """adapter should produce one tree for each route in the serialized data."""
        target_info = TargetInfo(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        trees = list(self.adapter.adapt(serialized_ibuprofen_data, target_info))
        assert len(trees) == len(serialized_ibuprofen_data)

    def test_adapt_one_step_route(self, serialized_ibuprofen_data):
        """correctly parses a single-step route."""
        # find the one-step route in the data
        one_step_route_data = next(r for r in serialized_ibuprofen_data if r["metadata"]["steps"] == 1)
        target_info = TargetInfo(id="ibuprofen", smiles=IBUPROFEN_SMILES)

        tree = next(self.adapter.adapt([one_step_route_data], target_info))
        root = tree.retrosynthetic_tree
        assert root.smiles == IBUPROFEN_SMILES

        reaction = root.reactions[0]
        assert len(reaction.reactants) == 2

        # derive expectations from the raw serialized data
        expected_reactants = {canonicalize_smiles(s) for s in one_step_route_data["reactions"][0]["reactants"]}
        actual_reactants = {r.smiles for r in reaction.reactants}

        assert actual_reactants == expected_reactants
        assert all(r.is_starting_material for r in reaction.reactants)

    def test_adapt_two_step_route(self, serialized_ibuprofen_data):
        """correctly parses a two-step route with a convergent step."""
        # find a specific two-step route
        two_step_route_data = serialized_ibuprofen_data[0]  # the first one is a 2-stepper
        assert two_step_route_data["metadata"]["steps"] == 2

        target_info = TargetInfo(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        tree = next(self.adapter.adapt([two_step_route_data], target_info))
        root = tree.retrosynthetic_tree

        # level 1
        reaction1 = root.reactions[0]
        intermediate_node = next(r for r in reaction1.reactants if not r.is_starting_material)

        # level 2
        reaction2 = intermediate_node.reactions[0]
        assert all(r.is_starting_material for r in reaction2.reactants)

        # derive expectations for L2 from the raw data
        l2_reaction_data = next(
            rxn
            for rxn in two_step_route_data["reactions"]
            if canonicalize_smiles(rxn["product"]) == intermediate_node.smiles
        )
        expected_reactants_l2 = {canonicalize_smiles(s) for s in l2_reaction_data["reactants"]}
        actual_reactants_l2 = {r.smiles for r in reaction2.reactants}

        assert actual_reactants_l2 == expected_reactants_l2

    def test_adapt_three_step_route(self, serialized_ibuprofen_data):
        """correctly parses a three-step route."""
        three_step_route_data = next(r for r in serialized_ibuprofen_data if r["metadata"]["steps"] == 3)
        target_info = TargetInfo(id="ibuprofen", smiles=IBUPROFEN_SMILES)

        tree = next(self.adapter.adapt([three_step_route_data], target_info))
        root = tree.retrosynthetic_tree

        # level 1
        reaction1 = root.reactions[0]
        intermediate1 = next(r for r in reaction1.reactants if not r.is_starting_material)

        # level 2
        reaction2 = intermediate1.reactions[0]
        intermediate2 = next(r for r in reaction2.reactants if not r.is_starting_material)

        # level 3
        reaction3 = intermediate2.reactions[0]
        assert all(r.is_starting_material for r in reaction3.reactants)
