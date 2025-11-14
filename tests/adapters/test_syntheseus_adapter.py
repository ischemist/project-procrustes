import pytest

from retrocast.adapters.syntheseus_adapter import SyntheseusAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.DEPRECATE_schemas import TargetInfo
from tests.adapters.test_base_adapter import BaseAdapterTest

# derive SMILES from the raw data to ensure canonicalization matches
PARACETAMOL_SMILES = canonicalize_smiles("CC(=O)Nc1ccc(O)cc1")
USPTO_2_SMILES = canonicalize_smiles("CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC")


class TestSyntheseusAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return SyntheseusAdapter()

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
class TestSyntheseusAdapterIntegration:
    adapter = SyntheseusAdapter()

    def test_adapt_multi_route_complex_target(self, raw_syntheseus_data):
        """tests a successful run with multiple, complex routes for a single target."""
        raw_data = raw_syntheseus_data["USPTO-2/190"]
        target_info = TargetInfo(id="USPTO-2/190", smiles=USPTO_2_SMILES)

        trees = list(self.adapter.adapt(raw_data, target_info))

        # the file contains 10 distinct routes for this target
        assert len(trees) == 10

        # --- deep inspection of the first tree ---
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert tree.target.id == "USPTO-2/190"
        assert root.smiles == USPTO_2_SMILES
        assert not root.is_starting_material
        assert len(root.reactions) == 1

        # check first step of decomposition
        reaction = root.reactions[0]
        assert len(reaction.reactants) == 1
        intermediate = reaction.reactants[0]
        assert not intermediate.is_starting_material

    def test_adapt_purchasable_target(self, raw_syntheseus_data):
        """tests a target that is purchasable, resulting in a 0-step route."""
        raw_data = raw_syntheseus_data["paracetamol"]
        target_info = TargetInfo(id="paracetamol", smiles=PARACETAMOL_SMILES)

        trees = list(self.adapter.adapt(raw_data, target_info))

        assert len(trees) == 1
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert root.smiles == PARACETAMOL_SMILES
        assert root.is_starting_material is True
        assert not root.reactions

    def test_adapt_no_routes_found(self, raw_syntheseus_data):
        """tests a target for which the model found no routes (empty list)."""
        raw_data = raw_syntheseus_data["ibuprofen"]
        target_info = TargetInfo(id="ibuprofen", smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1")

        trees = list(self.adapter.adapt(raw_data, target_info))
        assert len(trees) == 0
