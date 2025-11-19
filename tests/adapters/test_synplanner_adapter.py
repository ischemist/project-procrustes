import pytest

from retrocast.adapters.synplanner_adapter import SynPlannerAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

# ========================================
# Unit Tests (inherits from BaseAdapterTest)
# ========================================


class TestSynPlannerAdapterUnit(BaseAdapterTest):
    """Unit tests: verify adapter behavior with minimal fixtures."""

    @pytest.fixture
    def adapter_instance(self):
        return SynPlannerAdapter()

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
                        "smiles": "[C:1][C:2]=[O:3].[H:4][H:5]>>[C:1][C:2][O:3][H:4]",
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


# ========================================
# Contract Tests
# ========================================


@pytest.mark.contract
class TestSynPlannerAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SynPlannerAdapter:
        return SynPlannerAdapter()

    @pytest.fixture(scope="class")
    def paracetamol_routes(self, adapter: SynPlannerAdapter, raw_synplanner_data):
        """Shared fixture for paracetamol routes."""
        target = TargetInput(id="paracetamol", smiles=canonicalize_smiles("c1cc(ccc1O)NC(C)=O"))
        raw_routes = raw_synplanner_data["paracetamol"]
        return list(adapter.cast(raw_routes, target))

    def test_produces_correct_number_of_routes(self, paracetamol_routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(paracetamol_routes) == 14

    def test_all_routes_have_ranks(self, paracetamol_routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in paracetamol_routes]
        assert ranks == list(range(1, len(paracetamol_routes) + 1))

    def test_all_routes_are_valid_route_objects(self, paracetamol_routes):
        """Verify all routes are Route instances."""
        for route in paracetamol_routes:
            assert isinstance(route, Route)

    def test_all_routes_have_inchikeys(self, paracetamol_routes):
        """Verify all target molecules have InChIKeys."""
        for route in paracetamol_routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_non_leaf_molecules_have_synthesis_steps(self, paracetamol_routes):
        """Verify all non-leaf molecules have synthesis steps."""

        def check_molecule(mol):
            if not mol.is_leaf:
                assert mol.synthesis_step is not None
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in paracetamol_routes:
            check_molecule(route.target)

    def test_all_reaction_steps_have_mapped_smiles(self, paracetamol_routes):
        """Verify all reaction steps have mapped SMILES populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.mapped_smiles is not None
                assert len(mol.synthesis_step.mapped_smiles) > 0
                # Verify it contains atom mapping (has colons)
                assert ":" in mol.synthesis_step.mapped_smiles
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in paracetamol_routes:
            check_molecule(route.target)


# ========================================
# Regression Tests
# ========================================


@pytest.mark.regression
class TestSynPlannerAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SynPlannerAdapter:
        return SynPlannerAdapter()

    def test_aspirin_first_route_is_multi_step(self, adapter, raw_synplanner_data):
        """Verify the first aspirin route is a multi-step synthesis."""
        target = TargetInput(id="aspirin", smiles=canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        raw_routes = raw_synplanner_data["aspirin"]
        routes = list(adapter.cast(raw_routes, target))

        first_route = routes[0]
        assert first_route.rank == 1
        assert first_route.target.smiles == target.smiles
        assert not first_route.target.is_leaf

        # Check the route structure: aspirin has 2 reactants
        synthesis_step = first_route.target.synthesis_step
        assert synthesis_step is not None
        assert len(synthesis_step.reactants) == 2

        # One reactant is purchasable, one is an intermediate
        leaf_reactants = [r for r in synthesis_step.reactants if r.is_leaf]
        intermediate_reactants = [r for r in synthesis_step.reactants if not r.is_leaf]
        assert len(leaf_reactants) == 1
        assert len(intermediate_reactants) == 1

        # The leaf should be acetic acid
        assert leaf_reactants[0].smiles == canonicalize_smiles("O=C(O)C")

        # The intermediate should be salicylic acid (with further synthesis)
        intermediate = intermediate_reactants[0]
        assert intermediate.smiles == canonicalize_smiles("OC(=O)c1c(cccc1)O")
        assert intermediate.synthesis_step is not None

    def test_paracetamol_first_route_is_multi_step(self, adapter, raw_synplanner_data):
        """Verify the first paracetamol route has the expected multi-step structure."""
        target = TargetInput(id="paracetamol", smiles=canonicalize_smiles("c1cc(ccc1O)NC(C)=O"))
        raw_routes = raw_synplanner_data["paracetamol"]
        routes = list(adapter.cast(raw_routes, target))

        first_route = routes[0]
        assert first_route.rank == 1
        target = first_route.target

        # Verify target molecule
        assert target.smiles == target.smiles
        assert not target.is_leaf

        # Step 1: paracetamol synthesis
        step1 = target.synthesis_step
        assert step1 is not None
        assert len(step1.reactants) == 2

        # Find the intermediate (not in stock)
        intermediates = [r for r in step1.reactants if not r.is_leaf]
        assert len(intermediates) == 1
        intermediate1 = intermediates[0]

        # Step 2: intermediate synthesis
        step2 = intermediate1.synthesis_step
        assert step2 is not None
        assert len(step2.reactants) == 2

        # Find the next intermediate
        intermediates2 = [r for r in step2.reactants if not r.is_leaf]
        assert len(intermediates2) == 1
        intermediate2 = intermediates2[0]

        # Step 3: final intermediate synthesis
        step3 = intermediate2.synthesis_step
        assert step3 is not None
        assert len(step3.reactants) == 1

        # Final leaf should match expected SMILES
        final_leaf = step3.reactants[0]
        assert final_leaf.is_leaf
        assert final_leaf.smiles == canonicalize_smiles("CC(C)(C)Oc1ccccc1")

    def test_paracetamol_first_route_mapped_smiles(self, adapter, raw_synplanner_data):
        """Verify the mapped SMILES for reactions in the first paracetamol route."""
        target = TargetInput(id="paracetamol", smiles=canonicalize_smiles("c1cc(ccc1O)NC(C)=O"))
        raw_routes = raw_synplanner_data["paracetamol"]
        routes = list(adapter.cast(raw_routes, target))

        first_route = routes[0]

        # Step 1: paracetamol synthesis reaction
        step1 = first_route.target.synthesis_step
        assert step1 is not None
        assert (
            step1.mapped_smiles
            == "[O:3]=[C:2]([NH2:4])[CH3:1].[cH:7]1[cH:6][c:5]([cH:11][cH:10][c:8]1[OH:9])[Br:12]>>[cH:7]1[cH:6][c:5]([cH:11][cH:10][c:8]1[OH:9])[NH:4][C:2]([CH3:1])=[O:3]"
        )

        # Step 2: bromination reaction (second step in the synthesis)
        intermediates = [r for r in step1.reactants if not r.is_leaf]
        assert len(intermediates) == 1
        step2 = intermediates[0].synthesis_step
        assert step2 is not None
        assert (
            step2.mapped_smiles
            == "[Br-:12].[c:6]1[c:7][c:8]([c:10][c:11][c:5]1)[OH:9]>>[cH:7]1[cH:6][c:5]([cH:11][cH:10][c:8]1[OH:9])[Br:12]"
        )

        # Step 3: phenol deprotection reaction (third step in the synthesis)
        intermediates2 = [r for r in step2.reactants if not r.is_leaf]
        assert len(intermediates2) == 1
        step3 = intermediates2[0].synthesis_step
        assert step3 is not None
        assert (
            step3.mapped_smiles
            == "[CH3:14][C:13]([CH3:16])([CH3:15])[O:9][c:8]1[c:10][c:11][c:5][c:6][c:7]1>>[CH3:16][CH:13]([CH3:14])[CH3:15].[cH:6]1[cH:7][c:8]([cH:10][cH:11][cH:5]1)[OH:9]"
        )

    def test_aspirin_first_route_mapped_smiles(self, adapter, raw_synplanner_data):
        """Verify the mapped SMILES for reactions in the first aspirin route."""
        target = TargetInput(id="aspirin", smiles=canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        raw_routes = raw_synplanner_data["aspirin"]
        routes = list(adapter.cast(raw_routes, target))

        first_route = routes[0]

        # Step 1: aspirin acetylation reaction
        step1 = first_route.target.synthesis_step
        assert step1 is not None
        assert (
            step1.mapped_smiles
            == "[O:1]=[C:2]([OH:14])[CH3:3].[OH:13][C:11](=[O:12])[c:10]1[c:5]([cH:6][cH:7][cH:8][cH:9]1)[OH:4]>>[OH:13][C:11](=[O:12])[c:10]1[c:5]([O:4][C:2](=[O:1])[CH3:3])[cH:6][cH:7][cH:8][cH:9]1"
        )

        # Step 2: salicylic acid deprotection reaction
        intermediates = [r for r in step1.reactants if not r.is_leaf]
        assert len(intermediates) == 1
        step2 = intermediates[0].synthesis_step
        assert step2 is not None
        assert (
            step2.mapped_smiles
            == "[CH3:16][CH:15]([CH3:17])[O:13][C:11](=[O:12])[c:10]1[c:9][c:8][c:7][c:6][c:5]1[OH:4]>>[CH3:16][CH2:15][CH3:17].[OH:13][C:11](=[O:12])[c:10]1[c:5]([cH:6][cH:7][cH:8][cH:9]1)[OH:4]"
        )
