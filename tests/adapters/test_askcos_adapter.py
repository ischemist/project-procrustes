import copy
from typing import Any

import pytest

from retrocast.adapters.askcos_adapter import AskcosAdapter
from retrocast.schemas import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestAskcosAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return AskcosAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a minimal but valid askcos output with one pathway
        return {
            "results": {
                "uds": {
                    "node_dict": {
                        "CCO": {"smiles": "CCO", "id": "chem1", "type": "chemical", "terminal": True},
                        "CC(=O)O": {"smiles": "CC(=O)O", "id": "chem2", "type": "chemical", "terminal": True},
                        "CC(=O)OCC": {"smiles": "CC(=O)OCC", "id": "chem0", "type": "chemical", "terminal": False},
                        "CC(=O)O.CCO>>CC(=O)OCC": {
                            "smiles": "CC(=O)O.CCO>>CC(=O)OCC",
                            "id": "rxn1",
                            "type": "reaction",
                        },
                    },
                    "uuid2smiles": {
                        "00000000-0000-0000-0000-000000000000": "CC(=O)OCC",
                        "uuid-rxn": "CC(=O)O.CCO>>CC(=O)OCC",
                        "uuid-chem1": "CCO",
                        "uuid-chem2": "CC(=O)O",
                    },
                    "pathways": [
                        [
                            {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
                            {"source": "uuid-rxn", "target": "uuid-chem1"},
                            {"source": "uuid-rxn", "target": "uuid-chem2"},
                        ]
                    ],
                }
            }
        }

    @pytest.fixture
    def raw_unsuccessful_run_data(self, raw_valid_route_data: dict[str, Any]):
        # an empty pathways list
        modified_data = copy.deepcopy(raw_valid_route_data)
        modified_data["results"]["uds"]["pathways"] = []
        return modified_data

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # 'uds' key is missing
        return {"results": {}}

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="ethyl_acetate", smiles="CCOC(C)=O")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethyl_acetate", smiles="CCO")


@pytest.mark.integration
class TestAskcosAdapterIntegration:
    def test_adapt_methylacetate(self, raw_askcos_data: dict[str, Any], methylacetate_target_input: TargetInput):
        """
        tests the full processing of a valid askcos output for a single target.
        it verifies the total number of routes and the structure of specific simple
        and multi-step routes.
        """
        adapter = AskcosAdapter()
        raw_target_data = raw_askcos_data["methylacetate"]
        routes = list(adapter.adapt(raw_target_data, methylacetate_target_input))

        # the input file has 15 distinct pathways defined
        assert len(routes) == 15

        # verify metadata is present in all routes
        for route in routes:
            assert route.metadata is not None
            assert "total_iterations" in route.metadata
            assert "total_chemicals" in route.metadata
            assert "total_reactions" in route.metadata
            assert "total_templates" in route.metadata
            assert "total_paths" in route.metadata

        # --- check 1: a simple, one-step route ---
        # this corresponds to the first pathway in the raw data
        route1 = routes[0]
        assert route1.rank == 1
        target1 = route1.target
        assert target1.smiles == "COC(C)=O"
        assert not target1.is_leaf
        assert target1.synthesis_step is not None

        reaction1 = target1.synthesis_step
        assert reaction1.mapped_smiles == "Cl[C:3]([CH3:4])=[O:5].[CH3:1][OH:2]>>[CH3:1][O:2][C:3]([CH3:4])=[O:5]"
        assert len(reaction1.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction1.reactants}
        assert reactant_smiles == {"CC(=O)Cl", "CO"}
        assert all(r.is_leaf for r in reaction1.reactants)

        # --- check 2: a more complex, two-step route ---
        # this corresponds to the second pathway in the raw data
        route2 = routes[1]
        assert route2.rank == 2
        target2 = route2.target
        assert target2.smiles == "COC(C)=O"
        assert not target2.is_leaf
        assert target2.synthesis_step is not None

        # first retrosynthetic step
        reaction_step1 = target2.synthesis_step
        assert reaction_step1.mapped_smiles is not None  # Should have mapped smiles
        assert len(reaction_step1.reactants) == 2
        reactant_smiles_step1 = {r.smiles for r in reaction_step1.reactants}
        assert reactant_smiles_step1 == {"C=[N+]=[N-]", "CC(=O)O"}

        # find the intermediate and the starting material from step 1
        diazomethane_mol = next(r for r in reaction_step1.reactants if r.smiles == "C=[N+]=[N-]")
        acetic_acid_mol = next(r for r in reaction_step1.reactants if r.smiles == "CC(=O)O")
        assert not diazomethane_mol.is_leaf
        assert acetic_acid_mol.is_leaf

        # second retrosynthetic step (from diazomethane)
        assert diazomethane_mol.synthesis_step is not None
        reaction_step2 = diazomethane_mol.synthesis_step
        assert reaction_step2.mapped_smiles is not None  # Should have mapped smiles
        assert len(reaction_step2.reactants) == 1
        final_reactant = reaction_step2.reactants[0]
        assert final_reactant.smiles == "CN(N=O)C(N)=O"
        assert final_reactant.is_leaf

    @pytest.mark.parametrize(
        "key_to_remove, error_match",
        [
            pytest.param("COC(C)=O", "node data for smiles 'COC(C)=O' not found", id="chemical_smiles_missing"),
            pytest.param(
                "CC(=O)Cl.CO>>COC(C)=O",
                "node data for reaction 'CC(=O)Cl.CO>>COC(C)=O' not found",
                id="reaction_smiles_missing",
            ),
        ],
    )
    def test_logs_warning_on_inconsistent_nodedict(
        self, raw_askcos_data, methylacetate_target_input, key_to_remove, error_match, caplog
    ):
        """Tests resilience to inconsistencies in the node_dict mapping."""
        adapter = AskcosAdapter()
        raw_target_data = raw_askcos_data["methylacetate"]
        corrupted_data = copy.deepcopy(raw_target_data)
        corrupted_data["results"]["uds"]["node_dict"].pop(key_to_remove, None)

        # The adapter should still run and produce routes for the non-corrupted pathways
        routes = list(adapter.adapt(corrupted_data, methylacetate_target_input))

        # The key assertion: we produced FEWER routes than total pathways.
        total_pathways = len(raw_target_data["results"]["uds"]["pathways"])
        assert len(routes) < total_pathways
        assert error_match in caplog.text
