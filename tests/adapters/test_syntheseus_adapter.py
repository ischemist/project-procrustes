import pytest

from retrocast.adapters.syntheseus_adapter import SyntheseusAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetInput
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
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")


@pytest.mark.contract
class TestSyntheseusAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SyntheseusAdapter:
        return SyntheseusAdapter()

    @pytest.fixture(scope="class")
    def uspto_routes(self, adapter: SyntheseusAdapter, raw_syntheseus_data):
        """Shared fixture for USPTO-2/190 routes."""
        target = TargetInput(id="USPTO-2/190", smiles=USPTO_2_SMILES)
        raw_routes = raw_syntheseus_data["USPTO-2/190"]
        return list(adapter.cast(raw_routes, target))

    def test_produces_correct_number_of_routes(self, uspto_routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(uspto_routes) == 10

    def test_all_routes_have_ranks(self, uspto_routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in uspto_routes]
        assert ranks == list(range(1, len(uspto_routes) + 1))

    def test_all_routes_are_valid_route_objects(self, uspto_routes):
        """Verify all routes are Route instances."""
        for route in uspto_routes:
            assert isinstance(route, Route)

    def test_all_routes_have_inchikeys(self, uspto_routes):
        """Verify all target molecules have InChIKeys."""
        for route in uspto_routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_non_leaf_molecules_have_synthesis_steps(self, uspto_routes):
        """Verify all non-leaf molecules have synthesis steps."""

        def check_molecule(mol):
            if not mol.is_leaf:
                assert mol.synthesis_step is not None
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in uspto_routes:
            check_molecule(route.target)


@pytest.mark.regression
class TestSyntheseusAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SyntheseusAdapter:
        return SyntheseusAdapter()

    def test_adapt_multi_route_complex_target(self, adapter, raw_syntheseus_data):
        """Tests a successful run with multiple, complex routes for a single target."""
        raw_data = raw_syntheseus_data["USPTO-2/190"]
        target = TargetInput(id="USPTO-2/190", smiles=USPTO_2_SMILES)

        routes = list(adapter.cast(raw_data, target))

        # The file contains 10 distinct routes for this target
        assert len(routes) == 10

        # --- Deep inspection of the first route ---
        first_route = routes[0]
        target = first_route.target

        assert first_route.rank == 1
        assert target.smiles == USPTO_2_SMILES
        assert not target.is_leaf
        assert target.synthesis_step is not None

        # Check first step of decomposition
        synthesis_step = target.synthesis_step
        assert len(synthesis_step.reactants) == 1
        intermediate = synthesis_step.reactants[0]
        assert not intermediate.is_leaf

    def test_adapt_purchasable_target(self, adapter, raw_syntheseus_data):
        """Tests a target that is purchasable, resulting in a 0-step route."""
        raw_data = raw_syntheseus_data["paracetamol"]
        target = TargetInput(id="paracetamol", smiles=PARACETAMOL_SMILES)

        routes = list(adapter.cast(raw_data, target))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        assert route.rank == 1
        assert target.smiles == PARACETAMOL_SMILES
        assert target.is_leaf
        assert target.synthesis_step is None

    def test_adapt_no_routes_found(self, adapter, raw_syntheseus_data):
        """Tests a target for which the model found no routes (empty list)."""
        raw_data = raw_syntheseus_data["ibuprofen"]
        target = TargetInput(id="ibuprofen", smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1")

        routes = list(adapter.cast(raw_data, target))
        assert len(routes) == 0
