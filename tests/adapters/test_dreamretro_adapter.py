import pytest

from retrocast.adapters.dreamretro_adapter import DreamRetroAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

# ============================================================================
# Unit Tests: Test common failure modes
# ============================================================================


class TestDreamRetroAdapterUnit(BaseAdapterTest):
    """Unit tests for DreamRetroAdapter - tests common adapter failure modes."""

    @pytest.fixture
    def adapter_instance(self):
        return DreamRetroAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        return {"succ": True, "routes": "CCO>>CC=O.[H][H]"}

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return {"succ": False, "routes": "CCO>>CC=O.[H][H]"}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # the adapter logic, not pydantic, checks for dict and 'succ' key
        return "this is not a dict"

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")

    def test_parser_raises_on_invalid_step_format(self, adapter_instance):
        """the private parser method should raise an error for malformed steps."""
        bad_route_str = "CCO>CC=O.O"  # missing a ">"
        with pytest.raises(AdapterLogicError, match="invalid format near"):
            adapter_instance._parse_route_string(bad_route_str)


# ============================================================================
# Contract Tests: Verify all routes meet schema requirements
# ============================================================================


@pytest.mark.contract
class TestDreamRetroAdapterContract:
    """Contract tests - verify all routes meet schema requirements."""

    adapter = DreamRetroAdapter()

    @pytest.fixture(scope="class")
    def mirabegron_routes(self, raw_dreamretro_data):
        """Run adapter once for Mirabegron and reuse results."""
        raw_data = raw_dreamretro_data["Mirabegron"]
        raw_route_str = raw_data["routes"]
        product_smi_raw, _ = raw_route_str.split(">>")
        target_input = TargetInput(id="Mirabegron", smiles=canonicalize_smiles(product_smi_raw))
        return list(self.adapter.cast(raw_data, target_input))

    @pytest.fixture(scope="class")
    def anagliptin_routes(self, raw_dreamretro_data):
        """Run adapter once for Anagliptin and reuse results."""
        raw_data = raw_dreamretro_data["Anagliptin"]
        raw_route_str = raw_data["routes"]
        root_smi_raw, _, _ = raw_route_str.split(">>")[0].split("|")[0].partition(">")
        target_input = TargetInput(id="Anagliptin", smiles=canonicalize_smiles(root_smi_raw))
        return list(self.adapter.cast(raw_data, target_input))

    def test_route_has_required_fields(self, mirabegron_routes):
        """All routes must have required Route fields populated."""
        for route in mirabegron_routes:
            assert route.target is not None
            assert route.rank > 0
            assert route.metadata is not None

    def test_molecule_has_required_fields(self, mirabegron_routes):
        """All Molecule objects must have required fields populated."""
        for route in mirabegron_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                assert mol.smiles is not None
                assert mol.inchikey is not None
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)

    def test_reaction_step_has_required_fields(self, mirabegron_routes):
        """All ReactionStep objects must have required fields populated."""
        for route in mirabegron_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                if mol.synthesis_step:
                    step = mol.synthesis_step
                    assert step.reactants is not None
                    assert len(step.reactants) > 0
                    molecules.extend(step.reactants)

    def test_leaf_molecules_have_no_synthesis_step(self, mirabegron_routes):
        """Leaf molecules should not have a synthesis_step."""
        for route in mirabegron_routes:
            molecules = [route.target]
            while molecules:
                mol = molecules.pop()
                if mol.is_leaf:
                    assert mol.synthesis_step is None
                else:
                    assert mol.synthesis_step is not None
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)

    def test_route_metadata_contains_dreamretro_fields(self, mirabegron_routes):
        """Route metadata should contain DreamRetro-specific fields."""
        for route in mirabegron_routes:
            assert "expand_model_call" in route.metadata
            assert "value_model_call" in route.metadata
            assert "reaction_nodes_lens" in route.metadata
            assert "mol_nodes_lens" in route.metadata

    def test_multi_step_route_has_nested_structure(self, anagliptin_routes):
        """Multi-step routes should have nested ReactionStep structures."""
        for route in anagliptin_routes:
            # At least one intermediate (non-leaf) molecule should exist
            molecules = [route.target]
            has_intermediate = False
            while molecules:
                mol = molecules.pop()
                if not mol.is_leaf and mol.synthesis_step:
                    for reactant in mol.synthesis_step.reactants:
                        if not reactant.is_leaf:
                            has_intermediate = True
                            break
                if mol.synthesis_step:
                    molecules.extend(mol.synthesis_step.reactants)
            assert has_intermediate, "Multi-step route should have intermediate molecules"


# ============================================================================
# Regression Tests: Verify specific routes match expected values
# ============================================================================


@pytest.mark.regression
class TestDreamRetroAdapterRegression:
    """Regression tests - verify specific routes match exact expected values."""

    adapter = DreamRetroAdapter()

    def test_mirabegron_single_step_route(self, raw_dreamretro_data):
        """Verify exact structure of Mirabegron single-step route."""
        raw_data = raw_dreamretro_data["Mirabegron"]
        raw_route_str = raw_data["routes"]

        product_smi_raw, reactants_smi_raw = raw_route_str.split(">>")

        target_input = TargetInput(id="Mirabegron", smiles=canonicalize_smiles(product_smi_raw))
        routes = list(self.adapter.cast(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        root = route.target

        # Target molecule
        assert root.smiles == canonicalize_smiles(product_smi_raw)
        assert root.synthesis_step is not None
        assert route.rank == 1

        # Synthesis step
        reaction = root.synthesis_step
        reactant_smiles = {r.smiles for r in reaction.reactants}
        expected_reactants = {canonicalize_smiles(s) for s in reactants_smi_raw.split(".")}
        assert reactant_smiles == expected_reactants

        # Route metadata
        assert route.metadata["expand_model_call"] == raw_data["expand_model_call"]
        assert route.metadata["value_model_call"] == raw_data["value_model_call"]
        assert route.metadata["reaction_nodes_lens"] == raw_data["reaction_nodes_lens"]
        assert route.metadata["mol_nodes_lens"] == raw_data["mol_nodes_lens"]

    def test_anagliptin_multi_step_route(self, raw_dreamretro_data):
        """Verify exact structure of Anagliptin multi-step route."""
        raw_data = raw_dreamretro_data["Anagliptin"]
        raw_route_str = raw_data["routes"]
        root_smi_raw, _, _ = raw_route_str.split(">>")[0].split("|")[0].partition(">")
        target_input = TargetInput(id="Anagliptin", smiles=canonicalize_smiles(root_smi_raw))

        routes = list(self.adapter.cast(raw_data, target_input))
        assert len(routes) == 1
        root = routes[0].target

        # Level 1: target molecule
        assert root.smiles == canonicalize_smiles(root_smi_raw)
        assert root.synthesis_step is not None
        reaction1 = root.synthesis_step
        assert len(reaction1.reactants) == 2

        # Find intermediate and leaf nodes
        intermediate_node = next(r for r in reaction1.reactants if not r.is_leaf)
        leaf_node = next(r for r in reaction1.reactants if r.is_leaf)
        assert leaf_node.smiles == "N#C[C@@H]1CCCN1C(=O)CCl"

        # Level 2: intermediate decomposes further
        assert intermediate_node.synthesis_step is not None
        reaction2 = intermediate_node.synthesis_step
        assert len(reaction2.reactants) == 2
        assert all(r.is_leaf for r in reaction2.reactants)

        # Verify level 2 reactants
        reactant_smiles_l2 = {r.smiles for r in reaction2.reactants}
        _, l2_reactants_raw = raw_route_str.split("|")[1].split(">>")
        expected_reactants_l2 = {canonicalize_smiles(s) for s in l2_reactants_raw.split(".")}
        assert reactant_smiles_l2 == expected_reactants_l2

        # Route metadata
        assert routes[0].metadata["expand_model_call"] == raw_data["expand_model_call"]
        assert routes[0].metadata["value_model_call"] == raw_data["value_model_call"]
        assert routes[0].metadata["reaction_nodes_lens"] == raw_data["reaction_nodes_lens"]
        assert routes[0].metadata["mol_nodes_lens"] == raw_data["mol_nodes_lens"]
