import pytest

from retrocast.adapters.multistepttl_adapter import TtlRetroAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
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
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")


@pytest.mark.integration
class TestTtlRetroAdapterContract:
    """contract tests for ttlretro adapter verifying schema compliance."""

    adapter = TtlRetroAdapter()

    @pytest.fixture(scope="class")
    def serialized_ibuprofen_data(self, multistepttl_ibuprofen_dir) -> list[dict]:
        """serializes the ibuprofen pickle data once for all tests in this module."""
        data = serialize_multistepttl_directory(multistepttl_ibuprofen_dir)
        assert data is not None, "serialization failed for ibuprofen"
        return data

    @pytest.mark.filterwarnings("ignore:numpy.core.numeric is deprecated:DeprecationWarning")
    def test_all_routes_have_ranks(self, serialized_ibuprofen_data):
        """all routes should have non-zero rank values."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))
        assert all(route.rank > 0 for route in routes)

    def test_all_routes_have_target_molecules_with_inchikeys(self, serialized_ibuprofen_data):
        """all routes should have target molecules with inchikeys."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))
        assert all(route.target.inchikey is not None for route in routes)
        assert all(len(route.target.inchikey) > 0 for route in routes)

    def test_all_starting_materials_have_no_synthesis_step(self, serialized_ibuprofen_data):
        """starting materials should have no synthesis step."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))

        def check_molecule(mol):
            if mol.synthesis_step is None:
                return  # starting material - ok
            # has synthesis step - check all reactants recursively
            for reactant in mol.synthesis_step.reactants:
                check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_route_depth_calculation(self, serialized_ibuprofen_data):
        """route depth should match the number of steps in metadata."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))

        for route, raw_route in zip(routes, serialized_ibuprofen_data, strict=False):
            expected_steps = raw_route["metadata"]["steps"]
            assert route.length == expected_steps

    def test_all_molecules_have_inchikeys(self, serialized_ibuprofen_data):
        """all molecules in route should have inchikeys."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))

        def check_molecule(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step is not None:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)


@pytest.mark.integration
class TestTtlRetroAdapterRegression:
    """regression tests for ttlretro adapter verifying specific values."""

    adapter = TtlRetroAdapter()

    @pytest.fixture(scope="class")
    def serialized_ibuprofen_data(self, multistepttl_ibuprofen_dir) -> list[dict]:
        """serializes the ibuprofen pickle data once for all tests in this module."""
        data = serialize_multistepttl_directory(multistepttl_ibuprofen_dir)
        assert data is not None, "serialization failed for ibuprofen"
        return data

    @pytest.mark.filterwarnings("ignore:numpy.core.numeric is deprecated:DeprecationWarning")
    def test_adapt_parses_all_routes(self, serialized_ibuprofen_data):
        """adapter should produce one route for each route in the serialized data."""
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        routes = list(self.adapter.cast(serialized_ibuprofen_data, target_input))
        assert len(routes) == len(serialized_ibuprofen_data)

    def test_adapt_one_step_route(self, serialized_ibuprofen_data):
        """correctly parses a single-step route."""
        # find the one-step route in the data
        one_step_route_data = next(r for r in serialized_ibuprofen_data if r["metadata"]["steps"] == 1)
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)

        route = next(self.adapter.cast([one_step_route_data], target_input))
        target_mol = route.target
        assert target_mol.smiles == IBUPROFEN_SMILES

        synthesis_step = target_mol.synthesis_step
        assert synthesis_step is not None
        assert len(synthesis_step.reactants) == 2

        # derive expectations from the raw serialized data
        expected_reactants = {canonicalize_smiles(s) for s in one_step_route_data["reactions"][0]["reactants"]}
        actual_reactants = {r.smiles for r in synthesis_step.reactants}

        assert actual_reactants == expected_reactants
        assert all(r.synthesis_step is None for r in synthesis_step.reactants)

    def test_adapt_two_step_route(self, serialized_ibuprofen_data):
        """correctly parses a two-step route with a convergent step."""
        # find a specific two-step route
        two_step_route_data = serialized_ibuprofen_data[0]  # the first one is a 2-stepper
        assert two_step_route_data["metadata"]["steps"] == 2

        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)
        route = next(self.adapter.cast([two_step_route_data], target_input))
        target_mol = route.target

        # level 1
        step1 = target_mol.synthesis_step
        assert step1 is not None
        intermediate = next(r for r in step1.reactants if r.synthesis_step is not None)

        # level 2
        step2 = intermediate.synthesis_step
        assert step2 is not None
        assert all(r.synthesis_step is None for r in step2.reactants)

        # derive expectations for L2 from the raw data
        l2_reaction_data = next(
            rxn
            for rxn in two_step_route_data["reactions"]
            if canonicalize_smiles(rxn["product"]) == intermediate.smiles
        )
        expected_reactants_l2 = {canonicalize_smiles(s) for s in l2_reaction_data["reactants"]}
        actual_reactants_l2 = {r.smiles for r in step2.reactants}

        assert actual_reactants_l2 == expected_reactants_l2

    def test_adapt_three_step_route(self, serialized_ibuprofen_data):
        """correctly parses a three-step route."""
        three_step_route_data = next(r for r in serialized_ibuprofen_data if r["metadata"]["steps"] == 3)
        target_input = TargetInput(id="ibuprofen", smiles=IBUPROFEN_SMILES)

        route = next(self.adapter.cast([three_step_route_data], target_input))
        target_mol = route.target

        # level 1
        step1 = target_mol.synthesis_step
        assert step1 is not None
        intermediate1 = next(r for r in step1.reactants if r.synthesis_step is not None)

        # level 2
        step2 = intermediate1.synthesis_step
        assert step2 is not None
        intermediate2 = next(r for r in step2.reactants if r.synthesis_step is not None)

        # level 3
        step3 = intermediate2.synthesis_step
        assert step3 is not None
        assert all(r.synthesis_step is None for r in step3.reactants)
