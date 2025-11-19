import pytest

from retrocast.adapters.synllama_adapter import SynLlaMaAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestSynLlamaAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return SynLlaMaAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # a single-step route for acetone from two precursors
        return [{"synthesis_string": "CC(=O)O;C;R1;CC(=O)C"}]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # an empty list, from a target with no routes found.
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # wrong key in the dict, will fail pydantic validation.
        return [{"invalid_key": "..."}]

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="acetone", smiles="CC(C)=O")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="acetone", smiles="CCC")

    def test_adapt_parses_multi_step_route(self, adapter_instance):
        """
        Tests that the adapter correctly parses a multi-step synthesis string
        and builds a tree with the correct depth and structure.

        Route: C -> CC -> CCC
        """
        multi_step_string = "C;R1;CC;C;R2;CCC"
        raw_data = [{"synthesis_string": multi_step_string}]
        target_input = TargetInput(id="multi-step-test", smiles=canonicalize_smiles("CCC"))

        routes = list(adapter_instance.cast(raw_data, target_input))

        assert len(routes) == 1
        route = routes[0]
        target = route.target

        # Level 1: CCC -> CC + C
        assert target.smiles == "CCC"
        assert not target.is_leaf
        synthesis_step = target.synthesis_step
        assert synthesis_step is not None
        assert {r.smiles for r in synthesis_step.reactants} == {"CC", "C"}

        # Level 2: find the intermediate (CC) and check its decomposition
        intermediate_node = next(r for r in synthesis_step.reactants if r.smiles == "CC")
        leaf_node = next(r for r in synthesis_step.reactants if r.smiles == "C")

        assert not intermediate_node.is_leaf
        assert leaf_node.is_leaf  # C is a starting material in this route

        synthesis_step2 = intermediate_node.synthesis_step
        assert synthesis_step2 is not None
        assert {r.smiles for r in synthesis_step2.reactants} == {"C"}
        assert all(r.is_leaf for r in synthesis_step2.reactants)

    @pytest.mark.parametrize(
        "bad_string, error_match",
        [
            ("C;R1", "malformed route: template 'R1' has no product"),
            ("R1;C", "no reactants found for product 'C'"),
            ("", "synthesis string is empty."),
        ],
    )
    def test_parser_raises_on_invalid_string_format(self, adapter_instance, bad_string, error_match):
        """tests that the private parser method raises specific logic errors."""
        with pytest.raises(AdapterLogicError, match=error_match):
            adapter_instance._parse_synthesis_string(bad_string)


@pytest.mark.integration
class TestSynLlamaAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SynLlaMaAdapter:
        return SynLlaMaAdapter()

    @pytest.fixture(scope="class", params=["Conivaptan hydrochloride", "AGN-190205", "USPTO-165/190"])
    def routes(self, adapter, raw_synllama_data, request):
        """Shared fixture to avoid re-running adaptation for every test."""
        target_id = request.param
        raw_routes = raw_synllama_data[target_id]

        # derive the canonical target smiles from the raw data itself
        synthesis_str = raw_routes[0]["synthesis_string"]
        cleaned_parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
        product_smi_raw = cleaned_parts[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_input = TargetInput(id=target_id, smiles=target_smi_canon)
        return list(adapter.cast(raw_routes, target_input))

    def test_produces_at_least_one_route(self, routes):
        """Verify the adapter produces at least one route."""
        assert len(routes) >= 1

    def test_all_routes_have_ranks(self, routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in routes]
        assert ranks == list(range(1, len(routes) + 1))

    def test_all_routes_have_inchikeys(self, routes):
        """Verify all target molecules have InChIKeys."""
        for route in routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_molecules_have_inchikeys(self, routes):
        """Verify all molecules in the route tree have InChIKeys."""

        def check_molecule(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step is not None:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)

    def test_root_molecule_is_not_leaf(self, routes):
        """Verify target molecules have synthesis steps (non-leaf)."""
        for route in routes:
            assert not route.target.is_leaf
            assert route.target.synthesis_step is not None


@pytest.mark.integration
class TestSynLlamaAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> SynLlaMaAdapter:
        return SynLlaMaAdapter()

    @pytest.fixture(scope="class")
    def conivaptan_routes(self, adapter, raw_synllama_data):
        """Shared fixture for Conivaptan hydrochloride routes."""
        target_id = "Conivaptan hydrochloride"
        raw_routes = raw_synllama_data[target_id]

        synthesis_str = raw_routes[0]["synthesis_string"]
        cleaned_parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
        product_smi_raw = cleaned_parts[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_input = TargetInput(id=target_id, smiles=target_smi_canon)
        return list(adapter.cast(raw_routes, target_input))

    @pytest.fixture(scope="class")
    def agn_routes(self, adapter, raw_synllama_data):
        """Shared fixture for AGN-190205 routes."""
        target_id = "AGN-190205"
        raw_routes = raw_synllama_data[target_id]

        synthesis_str = raw_routes[0]["synthesis_string"]
        cleaned_parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
        product_smi_raw = cleaned_parts[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_input = TargetInput(id=target_id, smiles=target_smi_canon)
        return list(adapter.cast(raw_routes, target_input))

    @pytest.fixture(scope="class")
    def uspto_routes(self, adapter, raw_synllama_data):
        """Shared fixture for USPTO-165/190 routes."""
        target_id = "USPTO-165/190"
        raw_routes = raw_synllama_data[target_id]

        synthesis_str = raw_routes[0]["synthesis_string"]
        cleaned_parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
        product_smi_raw = cleaned_parts[-1]
        target_smi_canon = canonicalize_smiles(product_smi_raw)

        target_input = TargetInput(id=target_id, smiles=target_smi_canon)
        return list(adapter.cast(raw_routes, target_input))

    def test_conivaptan_first_route_has_rank_one(self, conivaptan_routes):
        """Verify Conivaptan first route has rank 1."""
        assert conivaptan_routes[0].rank == 1

    def test_conivaptan_first_route_target_smiles(self, conivaptan_routes):
        """Verify Conivaptan target SMILES is correct."""
        target = conivaptan_routes[0].target
        assert target.smiles == canonicalize_smiles(
            "Cc1nc2c([nH]1)-c1ccccc1N(C(=O)c1ccc(NC(=O)c3ccccc3-c3ccccc3)cc1)CC2"
        )

    def test_agn_first_route_has_rank_one(self, agn_routes):
        """Verify AGN-190205 first route has rank 1."""
        assert agn_routes[0].rank == 1

    def test_agn_first_route_target_smiles(self, agn_routes):
        """Verify AGN-190205 target SMILES is correct."""
        target = agn_routes[0].target
        assert target.smiles == canonicalize_smiles("CC1(C)CCC(C)(C)c2cc(C#Cc3ccc(C(=O)O)cc3)ccc21")

    def test_uspto_first_route_has_rank_one(self, uspto_routes):
        """Verify USPTO-165/190 first route has rank 1."""
        assert uspto_routes[0].rank == 1

    def test_uspto_first_route_target_smiles(self, uspto_routes):
        """Verify USPTO-165/190 target SMILES is correct."""
        target = uspto_routes[0].target
        assert target.smiles == canonicalize_smiles("CCOC(=O)CCc1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2[nH]1")
