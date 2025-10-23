import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.aizynth_adapter import AizynthAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo


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
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
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

        trees = list(adapter.adapt(raw_routes, target_info))

        assert len(trees) == 11
        first_tree = trees[0]

        # check root node
        assert first_tree.target.id == "aspirin"
        assert first_tree.retrosynthetic_tree.smiles == target_info.smiles
        assert not first_tree.retrosynthetic_tree.is_starting_material

        # check the first, simplest route: salicylic acid + acetic anhydride -> aspirin
        reaction = first_tree.retrosynthetic_tree.reactions[0]
        assert len(reaction.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(=O)OC(C)=O"),  # acetic anhydride
            canonicalize_smiles("O=C(O)c1ccccc1O"),  # salicylic acid
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_starting_material for r in reaction.reactants)

    def test_aizynth_adapter_ibuprofen(self, adapter, raw_aizynth_mcts_data):
        """tests that the adapter correctly processes a multi-step ibuprofen route."""
        target_info = TargetInfo(id="ibuprofen", smiles=canonicalize_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"))
        # let's test against the first route provided in the file
        raw_route = raw_aizynth_mcts_data["ibuprofen"][0]

        trees = list(adapter.adapt([raw_route], target_info))
        assert len(trees) == 1
        tree = trees[0].retrosynthetic_tree

        # follow the path down:
        # ibuprofen -> intermediate 1 -> intermediate 2 -> starting materials
        assert tree.smiles == target_info.smiles
        assert not tree.is_starting_material
        # step 1
        rxn1 = tree.reactions[0]
        intermediate1 = rxn1.reactants[0]
        assert intermediate1.smiles == canonicalize_smiles("CC(C)C(=O)c1ccc([C@@H](C)C(=O)O)cc1")
        assert not intermediate1.is_starting_material
        # step 2
        rxn2 = intermediate1.reactions[0]
        intermediate2 = rxn2.reactants[0]
        assert intermediate2.smiles == canonicalize_smiles("COC(=O)[C@H](C)c1ccc(C(=O)C(C)C)cc1")
        assert not intermediate2.is_starting_material
        # step 3 (final)
        rxn3 = intermediate2.reactions[0]
        assert len(rxn3.reactants) == 2
        reactant_smiles = {r.smiles for r in rxn3.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(C)C(=O)O"),
            canonicalize_smiles("COC(=O)[C@H](C)c1ccccc1"),
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_starting_material for r in rxn3.reactants)
