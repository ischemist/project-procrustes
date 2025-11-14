import pytest

from retrocast.adapters.aizynth_adapter import AizynthAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.DEPRECATE_schemas import TargetInfo
from retrocast.schemas import Route
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestAizynthAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return AizynthAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        return [
            {
                "smiles": "CCO",
                "type": "mol",
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "CC=O.[H][H]>>CCO",
                        "children": [
                            {"smiles": "CC=O", "type": "mol", "in_stock": True, "children": []},
                            {"smiles": "[H][H]", "type": "mol", "in_stock": True, "children": []},
                        ],
                    }
                ],
            }
        ]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # 'type' is missing, which will fail pydantic discriminated union validation
        return [{"smiles": "CCO", "children": []}]

    @pytest.fixture
    def target_input(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInfo(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestAizynthAdapterIntegration:
    @pytest.fixture(scope="class")
    def adapter(self) -> AizynthAdapter:
        return AizynthAdapter()

    def test_aizynth_adapter_aspirin(self, adapter, raw_aizynth_mcts_data):
        """tests that the adapter correctly processes aspirin routes from aizynth mcts."""
        target_info = TargetInfo(id="aspirin", smiles=canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        raw_routes = raw_aizynth_mcts_data["aspirin"]

        # sanity check the input data
        assert len(raw_routes) == 11

        routes = list(adapter.adapt(raw_routes, target_info))

        assert len(routes) == 11
        first_route = routes[0]

        # check route and target molecule
        assert isinstance(first_route, Route)
        assert first_route.rank == 1
        assert first_route.target.smiles == target_info.smiles
        assert first_route.target.inchikey
        assert not first_route.target.is_leaf

        # check the first, simplest route: salicylic acid + acetic anhydride -> aspirin
        synthesis_step = first_route.target.synthesis_step
        assert synthesis_step is not None
        assert len(synthesis_step.reactants) == 2

        # check that template and mapped_smiles are extracted
        assert synthesis_step.template is not None
        assert synthesis_step.mapped_smiles is not None

        reactant_smiles = {r.smiles for r in synthesis_step.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(=O)OC(C)=O"),  # acetic anhydride
            canonicalize_smiles("O=C(O)c1ccccc1O"),  # salicylic acid
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_leaf for r in synthesis_step.reactants)

    def test_aizynth_adapter_ibuprofen(self, adapter, raw_aizynth_mcts_data):
        """tests that the adapter correctly processes a multi-step ibuprofen route."""
        target_info = TargetInfo(id="ibuprofen", smiles=canonicalize_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"))
        # let's test against the first route provided in the file
        raw_route = raw_aizynth_mcts_data["ibuprofen"][0]

        routes = list(adapter.adapt([raw_route], target_info))
        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # follow the path down:
        # ibuprofen -> intermediate 1 -> intermediate 2 -> starting materials
        assert target.smiles == target_info.smiles
        assert not target.is_leaf
        # step 1
        step1 = target.synthesis_step
        assert step1 is not None
        assert step1.template is not None  # check template is extracted
        assert step1.mapped_smiles is not None  # check mapped_smiles is extracted
        intermediate1 = step1.reactants[0]
        assert intermediate1.smiles == canonicalize_smiles("CC(C)C(=O)c1ccc([C@@H](C)C(=O)O)cc1")
        assert not intermediate1.is_leaf
        # step 2
        step2 = intermediate1.synthesis_step
        assert step2 is not None
        assert step2.template is not None
        assert step2.mapped_smiles is not None
        intermediate2 = step2.reactants[0]
        assert intermediate2.smiles == canonicalize_smiles("COC(=O)[C@H](C)c1ccc(C(=O)C(C)C)cc1")
        assert not intermediate2.is_leaf
        # step 3 (final)
        step3 = intermediate2.synthesis_step
        assert step3 is not None
        assert step3.template is not None
        assert step3.mapped_smiles is not None
        assert len(step3.reactants) == 2
        reactant_smiles = {r.smiles for r in step3.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(C)C(=O)O"),
            canonicalize_smiles("COC(=O)[C@H](C)c1ccccc1"),
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_leaf for r in step3.reactants)
