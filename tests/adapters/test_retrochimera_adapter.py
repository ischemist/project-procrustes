import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.retrochimera_adapter import RetrochimeraAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo


class TestRetrochimeraAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return RetrochimeraAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a minimal but valid retrochimera output
        return {
            "smiles": "CCO",
            "result": {
                "outputs": [
                    {
                        "routes": [
                            {
                                "reactions": [{"product": "CCO", "reactants": ["CC=O", "[H][H]"], "probability": 0.9}],
                                "num_steps": 1,
                                "step_probability_min": 0.9,
                                "step_probability_product": 0.9,
                            }
                        ],
                        "num_routes": 1,
                    }
                ]
            },
        }

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # represents a model failure
        return {"smiles": "CCO", "result": {"error": {"message": "failed"}}}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # 'result' key is missing, will fail pydantic validation
        return {"smiles": "CCO"}

    @pytest.fixture
    def target_info(self):
        return TargetInfo(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_info(self):
        return TargetInfo(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestRetrochimeraAdapterIntegration:
    adapter = RetrochimeraAdapter()

    def test_adapt_successful_routes(self, raw_retrochimera_data):
        """tests that the adapter correctly parses multiple valid routes from the input."""
        raw_data = raw_retrochimera_data["Ebastine"]

        # derive the target smiles directly from the raw data file
        target_smi_raw = raw_data["smiles"]
        target_info = TargetInfo(id="Ebastine", smiles=canonicalize_smiles(target_smi_raw))

        trees = list(self.adapter.adapt(raw_data, target_info))

        assert len(trees) == 3

        # --- deep inspection of the first route ---
        tree = trees[0]
        root = tree.retrosynthetic_tree

        assert tree.target.id == "Ebastine"
        assert root.smiles == canonicalize_smiles(target_smi_raw)
        assert not root.is_starting_material
        assert len(root.reactions) == 1

        reaction = root.reactions[0]
        assert len(reaction.reactants) == 2

        # derive the expected reactants from the raw data, then canonicalize them
        expected_reactants_raw = raw_data["result"]["outputs"][0]["routes"][0]["reactions"][0]["reactants"]
        expected_reactants_canon = {canonicalize_smiles(s) for s in expected_reactants_raw}

        # get the actual reactants from the parsed tree
        actual_reactants_canon = {r.smiles for r in reaction.reactants}

        assert actual_reactants_canon == expected_reactants_canon
        assert all(r.is_starting_material for r in reaction.reactants)
