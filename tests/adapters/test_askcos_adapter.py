import copy
from typing import Any

import pytest

from retrocast.adapters.askcos_adapter import AskcosAdapter
from retrocast.models.chem import TargetInput
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


@pytest.mark.contract
class TestAskcosAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AskcosAdapter:
        return AskcosAdapter()

    @pytest.fixture(scope="class")
    def routes(self, adapter: AskcosAdapter, raw_askcos_data: dict[str, Any], methylacetate_target_input: TargetInput):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_askcos_data["methylacetate"]
        return list(adapter.cast(raw_target_data, methylacetate_target_input))

    def test_produces_correct_number_of_routes(self, routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(routes) == 15

    def test_all_routes_have_metadata(self, routes):
        """Verify all routes have metadata with required fields."""
        for route in routes:
            assert route.metadata is not None
            assert "total_iterations" in route.metadata
            assert "total_chemicals" in route.metadata
            assert "total_reactions" in route.metadata
            assert "total_templates" in route.metadata
            assert "total_paths" in route.metadata

    def test_all_routes_have_ranks(self, routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in routes]
        assert ranks == list(range(1, len(routes) + 1))

    def test_all_routes_have_inchikeys(self, routes):
        """Verify all target molecules have InChIKeys."""
        for route in routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_non_leaf_molecules_have_synthesis_steps(self, routes):
        """Verify all non-leaf molecules have synthesis steps."""

        def check_molecule(mol):
            if not mol.is_leaf:
                assert mol.synthesis_step is not None
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_all_reaction_steps_have_mapped_smiles(self, routes):
        """Verify all reaction steps have mapped SMILES populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.mapped_smiles is not None
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_all_reaction_steps_have_templates(self, routes):
        """Verify all reaction steps have templates populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.template is not None
                assert len(mol.synthesis_step.template) > 0
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)


@pytest.mark.regression
class TestAskcosAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AskcosAdapter:
        return AskcosAdapter()

    @pytest.fixture(scope="class")
    def routes(self, adapter: AskcosAdapter, raw_askcos_data: dict[str, Any], methylacetate_target_input: TargetInput):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_askcos_data["methylacetate"]
        return list(adapter.cast(raw_target_data, methylacetate_target_input))

    def test_first_route_is_simple_one_step(self, routes):
        """Verify the first route is a simple one-step synthesis."""
        route1 = routes[0]
        assert route1.rank == 1

        target = route1.target
        assert target.smiles == "COC(C)=O"
        assert not target.is_leaf
        assert target.synthesis_step is not None

        reaction = target.synthesis_step
        assert len(reaction.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction.reactants}
        assert reactant_smiles == {"CC(=O)Cl", "CO"}
        assert all(r.is_leaf for r in reaction.reactants)

    def test_first_route_mapped_smiles(self, routes):
        """Verify the mapped SMILES for the first route matches expected value."""
        route1 = routes[0]
        reaction = route1.target.synthesis_step
        assert reaction.mapped_smiles == "Cl[C:3]([CH3:4])=[O:5].[CH3:1][OH:2]>>[CH3:1][O:2][C:3]([CH3:4])=[O:5]"

    def test_first_route_template(self, routes):
        """Verify the template for the first route contains expected pattern."""
        route1 = routes[0]
        reaction = route1.target.synthesis_step
        assert reaction.template is not None
        # Template should be a SMARTS pattern with reaction arrow
        assert ">>" in reaction.template

    def test_second_route_is_two_step(self, routes):
        """Verify the second route is a two-step synthesis."""
        route2 = routes[1]
        assert route2.rank == 2

        target = route2.target
        assert target.smiles == "COC(C)=O"
        assert not target.is_leaf

        # First step reactants
        step1 = target.synthesis_step
        assert len(step1.reactants) == 2
        reactant_smiles_step1 = {r.smiles for r in step1.reactants}
        assert reactant_smiles_step1 == {"C=[N+]=[N-]", "CC(=O)O"}

        # Find intermediate and starting material
        diazomethane = next(r for r in step1.reactants if r.smiles == "C=[N+]=[N-]")
        acetic_acid = next(r for r in step1.reactants if r.smiles == "CC(=O)O")

        assert not diazomethane.is_leaf
        assert acetic_acid.is_leaf

        # Second step
        step2 = diazomethane.synthesis_step
        assert step2 is not None
        assert len(step2.reactants) == 1
        assert step2.reactants[0].smiles == "CN(N=O)C(N)=O"
        assert step2.reactants[0].is_leaf


@pytest.mark.contract
class TestAskcosAdapterErrorHandling:
    """Tests for error handling and edge cases."""

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
        routes = list(adapter.cast(corrupted_data, methylacetate_target_input))

        # The key assertion: we produced FEWER routes than total pathways.
        total_pathways = len(raw_target_data["results"]["uds"]["pathways"])
        assert len(routes) < total_pathways
        assert error_match in caplog.text
