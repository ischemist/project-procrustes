from __future__ import annotations

import pytest

from retrocast.adapters.retrochimera_adapter import RetrochimeraAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest


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
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestRetrochimeraAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> RetrochimeraAdapter:
        return RetrochimeraAdapter()

    @pytest.fixture(scope="class")
    def ebastine_target_input(self) -> TargetInput:
        """Provides the target input for Ebastine from raw data."""
        return TargetInput(
            id="Ebastine",
            smiles=canonicalize_smiles("CC(C)(C)C1=CC=C(C=C1)C(=O)CCCN2CCC(CC2)OC(C3=CC=CC=C3)C4=CC=CC=C4"),
        )

    @pytest.fixture(scope="class")
    def routes(self, adapter, raw_retrochimera_data, ebastine_target_input):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_retrochimera_data["Ebastine"]
        return list(adapter.cast(raw_target_data, ebastine_target_input))

    def test_produces_correct_number_of_routes(self, routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(routes) == 3

    def test_all_routes_have_ranks(self, routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in routes]
        assert ranks == list(range(1, len(routes) + 1))

    def test_all_routes_have_inchikeys(self, routes):
        """Verify all target molecules have InChIKeys."""
        for route in routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_targets_are_non_leaf(self, routes):
        """Verify all target molecules have synthesis steps (not leaves)."""
        for route in routes:
            assert not route.target.is_leaf
            assert route.target.synthesis_step is not None

    def test_all_reactants_are_leaves(self, routes):
        """Verify all immediate reactants are leaf nodes."""
        for route in routes:
            if route.target.synthesis_step:
                for reactant in route.target.synthesis_step.reactants:
                    assert reactant.is_leaf


@pytest.mark.integration
class TestRetrochimeraAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> RetrochimeraAdapter:
        return RetrochimeraAdapter()

    @pytest.fixture(scope="class")
    def ebastine_target_input(self) -> TargetInput:
        """Provides the target input for Ebastine from raw data."""
        return TargetInput(
            id="Ebastine",
            smiles=canonicalize_smiles("CC(C)(C)C1=CC=C(C=C1)C(=O)CCCN2CCC(CC2)OC(C3=CC=CC=C3)C4=CC=CC=C4"),
        )

    @pytest.fixture(scope="class")
    def routes(self, adapter, raw_retrochimera_data, ebastine_target_input):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_retrochimera_data["Ebastine"]
        return list(adapter.cast(raw_target_data, ebastine_target_input))

    def test_first_route_has_correct_target(self, routes, ebastine_target_input):
        """Verify the first route has the correct target SMILES."""
        route1 = routes[0]
        assert route1.rank == 1
        assert route1.target.smiles == ebastine_target_input.smiles
        assert not route1.target.is_leaf

    def test_first_route_has_synthesis_step(self, routes):
        """Verify the first route's target has a synthesis step."""
        route1 = routes[0]
        assert route1.target.synthesis_step is not None
        assert len(route1.target.synthesis_step.reactants) == 2

    def test_first_route_reactants_are_starting_materials(self, routes):
        """Verify all reactants in the first route are starting materials."""
        route1 = routes[0]
        reaction = route1.target.synthesis_step
        assert reaction is not None
        assert all(r.is_leaf for r in reaction.reactants)

    def test_routes_have_distinct_structures(self, routes):
        """Verify that routes have different structures (different signatures)."""
        signatures = [route.get_signature() for route in routes]
        # All 3 routes should be distinct
        assert len(set(signatures)) == 3

    def test_all_molecules_have_canonical_smiles(self, routes):
        """Verify all molecules have canonical SMILES strings."""

        def check_molecule(mol):
            # The SMILES should match when canonicalized again
            assert canonicalize_smiles(mol.smiles) == mol.smiles
            if mol.synthesis_step:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_all_molecules_have_inchikeys(self, routes):
        """Verify all molecules in all routes have InChIKeys."""

        def check_molecule(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)
