import pytest

from retrocast.adapters.aizynth_adapter import AizynthAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetInput
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
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")


@pytest.mark.contract
class TestAizynthAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AizynthAdapter:
        return AizynthAdapter()

    @pytest.fixture(scope="class")
    def aspirin_routes(self, adapter: AizynthAdapter, raw_aizynth_mcts_data):
        """Shared fixture for aspirin routes."""
        target = TargetInput(id="aspirin", smiles=canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        raw_routes = raw_aizynth_mcts_data["aspirin"]
        return list(adapter.cast(raw_routes, target))

    def test_produces_correct_number_of_routes(self, aspirin_routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(aspirin_routes) == 11

    def test_all_routes_have_ranks(self, aspirin_routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in aspirin_routes]
        assert ranks == list(range(1, len(aspirin_routes) + 1))

    def test_all_routes_are_valid_route_objects(self, aspirin_routes):
        """Verify all routes are Route instances."""
        for route in aspirin_routes:
            assert isinstance(route, Route)

    def test_all_routes_have_inchikeys(self, aspirin_routes):
        """Verify all target molecules have InChIKeys."""
        for route in aspirin_routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_non_leaf_molecules_have_synthesis_steps(self, aspirin_routes):
        """Verify all non-leaf molecules have synthesis steps."""

        def check_molecule(mol):
            if not mol.is_leaf:
                assert mol.synthesis_step is not None
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in aspirin_routes:
            check_molecule(route.target)

    def test_all_reaction_steps_have_templates(self, aspirin_routes):
        """Verify all reaction steps have templates populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.template is not None
                assert len(mol.synthesis_step.template) > 0
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in aspirin_routes:
            check_molecule(route.target)

    def test_all_reaction_steps_have_mapped_smiles(self, aspirin_routes):
        """Verify all reaction steps have mapped SMILES populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.mapped_smiles is not None
                assert len(mol.synthesis_step.mapped_smiles) > 0
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in aspirin_routes:
            check_molecule(route.target)


@pytest.mark.regression
class TestAizynthAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AizynthAdapter:
        return AizynthAdapter()

    def test_aspirin_first_route_is_one_step(self, adapter, raw_aizynth_mcts_data):
        """Verify the first aspirin route is a simple one-step synthesis."""
        target = TargetInput(id="aspirin", smiles=canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        raw_routes = raw_aizynth_mcts_data["aspirin"]
        routes = list(adapter.cast(raw_routes, target))

        first_route = routes[0]
        assert first_route.rank == 1
        assert first_route.target.smiles == target.smiles
        assert not first_route.target.is_leaf

        # Check the route: salicylic acid + acetic anhydride -> aspirin
        synthesis_step = first_route.target.synthesis_step
        assert synthesis_step is not None
        assert len(synthesis_step.reactants) == 2

        reactant_smiles = {r.smiles for r in synthesis_step.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(=O)OC(C)=O"),  # acetic anhydride
            canonicalize_smiles("O=C(O)c1ccccc1O"),  # salicylic acid
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_leaf for r in synthesis_step.reactants)

    def test_ibuprofen_first_route_is_three_step(self, adapter, raw_aizynth_mcts_data):
        """Verify the first ibuprofen route is a three-step synthesis."""
        target = TargetInput(id="ibuprofen", smiles=canonicalize_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"))
        raw_route = raw_aizynth_mcts_data["ibuprofen"][0]
        routes = list(adapter.cast([raw_route], target))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Verify three-step path: ibuprofen -> intermediate 1 -> intermediate 2 -> starting materials
        assert target.smiles == target.smiles
        assert not target.is_leaf

        # Step 1
        step1 = target.synthesis_step
        assert step1 is not None
        intermediate1 = step1.reactants[0]
        assert intermediate1.smiles == canonicalize_smiles("CC(C)C(=O)c1ccc([C@@H](C)C(=O)O)cc1")
        assert not intermediate1.is_leaf

        # Step 2
        step2 = intermediate1.synthesis_step
        assert step2 is not None
        intermediate2 = step2.reactants[0]
        assert intermediate2.smiles == canonicalize_smiles("COC(=O)[C@H](C)c1ccc(C(=O)C(C)C)cc1")
        assert not intermediate2.is_leaf

        # Step 3 (final)
        step3 = intermediate2.synthesis_step
        assert step3 is not None
        assert len(step3.reactants) == 2
        reactant_smiles = {r.smiles for r in step3.reactants}
        expected_smiles = {
            canonicalize_smiles("CC(C)C(=O)O"),
            canonicalize_smiles("COC(=O)[C@H](C)c1ccccc1"),
        }
        assert reactant_smiles == expected_smiles
        assert all(r.is_leaf for r in step3.reactants)
