from copy import deepcopy

import pytest

from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest


class TestPaRoutesAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return PaRoutesAdapter()

    @pytest.fixture
    def raw_valid_route_data(self, raw_paroutes_data):
        # the `adapt` method receives the raw data for a single target, which is a dict.
        return raw_paroutes_data["paroutes-ex-1"]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # an "unsuccessful" run for a target might be an empty dict, which will fail validation.
        # the adapter should yield nothing, which is correct.
        return {}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # missing 'type' will fail pydantic validation.
        return {"smiles": "CCO", "children": []}

    @pytest.fixture
    def target_input(self, raw_paroutes_data):
        smiles = raw_paroutes_data["paroutes-ex-1"]["smiles"]
        return TargetInput(id="paroutes-ex-1", smiles=canonicalize_smiles(smiles))

    @pytest.fixture
    def mismatched_target_input(self, raw_paroutes_data):
        return TargetInput(id="paroutes-ex-1", smiles="CCO")  # clearly not the same molecule


@pytest.mark.integration
class TestPaRoutesAdapterContract:
    """contract tests: verify the adapter produces valid route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> PaRoutesAdapter:
        return PaRoutesAdapter()

    @pytest.fixture(scope="class")
    def routes_ex1(self, adapter, raw_paroutes_data):
        """shared fixture to avoid re-running adaptation for every test."""
        target_id = "paroutes-ex-1"
        raw_route = raw_paroutes_data[target_id]
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))
        return list(adapter.cast(raw_route, target_input))

    @pytest.fixture(scope="class")
    def routes_ex2(self, adapter, raw_paroutes_data):
        """shared fixture for second example."""
        target_id = "paroutes-ex-2"
        raw_route = raw_paroutes_data[target_id]
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))
        return list(adapter.cast(raw_route, target_input))

    def test_produces_single_route(self, routes_ex1):
        """verify the adapter produces exactly one route per target."""
        assert len(routes_ex1) == 1

    def test_route_has_rank(self, routes_ex1):
        """verify the route has rank 1."""
        assert routes_ex1[0].rank == 1

    def test_route_has_patent_id_metadata(self, routes_ex1):
        """verify the route metadata contains patent_id."""
        route = routes_ex1[0]
        assert "patent_id" in route.metadata
        assert route.metadata["patent_id"] == "US20150051201A1"

    def test_all_molecules_have_inchikeys(self, routes_ex1):
        """verify all molecules in the route have inchikeys."""

        def check_molecule(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step is not None:
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        check_molecule(routes_ex1[0].target)

    def test_all_reaction_steps_have_mapped_smiles(self, routes_ex1):
        """verify all reaction steps have mapped smiles (rsmi) populated."""

        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.mapped_smiles is not None
                assert len(mol.synthesis_step.mapped_smiles) > 0
                # verify it contains atom mapping (colon followed by digit)
                assert ":" in mol.synthesis_step.mapped_smiles
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        check_molecule(routes_ex1[0].target)

    def test_length_calculation(self, routes_ex1, routes_ex2):
        """verify route length is calculated correctly."""
        # paroutes-ex-1 has 2 reaction steps (length 2)
        assert routes_ex1[0].length == 2
        # paroutes-ex-2 has 3 reaction steps (length 3)
        assert routes_ex2[0].length == 3


@pytest.mark.integration
class TestPaRoutesAdapterRegression:
    """regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> PaRoutesAdapter:
        return PaRoutesAdapter()

    def test_adapt_valid_single_patent_route(self, adapter, raw_paroutes_data):
        """
        tests that a route where all reaction steps are from the same patent
        is successfully adapted.
        """
        target_id = "paroutes-ex-1"
        raw_route = raw_paroutes_data[target_id]
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # both reaction steps in this example are from patent 'us20150051201a1'.
        routes = list(adapter.cast(raw_route, target_input))

        assert len(routes) == 1
        route = routes[0]
        assert route.target.smiles == target_input.smiles
        assert not route.target.is_leaf
        # check that it has some depth
        reaction = route.target.synthesis_step
        assert reaction is not None
        assert len(reaction.reactants) == 2
        # check one level deeper
        intermediate_mol = next(r for r in reaction.reactants if not r.is_leaf)
        assert intermediate_mol.synthesis_step is not None

    def test_rejects_mixed_patent_route(self, adapter, raw_paroutes_data):
        """
        tests that a route is REJECTED if its reaction steps come from
        different patents. this is the key custom logic for this adapter.
        """
        target_id = "paroutes-ex-1"
        # use deepcopy to avoid state leakage between tests
        raw_route = deepcopy(raw_paroutes_data[target_id])
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # let's mutate the data to create the failure condition.
        # the first reaction id is 'us20150051201a1;0516;1654836'
        # we'll change the second one.
        # path: children[0] (reaction) -> children[1] (mol) -> children[0] (reaction)
        inner_reaction = raw_route["children"][0]["children"][1]["children"][0]
        inner_reaction["metadata"]["ID"] = "SOME-OTHER-PATENT;1234;56789"

        # the adapter should now see two different patent ids and yield nothing.
        routes = list(adapter.cast(raw_route, target_input))

        assert len(routes) == 0

    def test_adapt_second_example_route(self, adapter, raw_paroutes_data):
        """
        tests adaptation on the second example to ensure robustness.
        """
        target_id = "paroutes-ex-2"
        raw_route = raw_paroutes_data[target_id]
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # all reaction steps in this example are from patent 'us08242133b2'.
        routes = list(adapter.cast(raw_route, target_input))

        assert len(routes) == 1
        route = routes[0]
        assert route.target.smiles == target_input.smiles

        # just check the first reaction's children
        reaction1 = route.target.synthesis_step
        assert reaction1 is not None
        assert len(reaction1.reactants) == 2
        reactant_smiles = {r.smiles for r in reaction1.reactants}
        expected_smiles = {
            canonicalize_smiles("Nc1cc(OC(F)(F)F)ccc1O"),
            canonicalize_smiles("O=C(O)c1ccncc1Cl"),
        }
        assert reactant_smiles == expected_smiles


class TestPaRoutesYearParsing:
    @pytest.mark.parametrize(
        "patent_id, expected_year, expected_category, expected_cat_count",
        [
            # --- Correctly Parsed Modern Application IDs ---
            ("US20150051201A1", "2015", None, 0),
            ("US20011234567B2", "2001", None, 0),
            ("US20999999999A1", "2099", None, 0),
            # --- Pre-2001 Granted Patents (No Year Info) ---
            ("US6039312B1", None, "pre-2001_grant", 1),
            ("US0940123A1", None, "pre-2001_grant", 1),
            # This would be an invalid ID, but tests the digit-first logic
            ("US19991234567A1", None, "pre-2001_grant", 1),
            # --- Special/Administrative Patents ---
            ("USRE037303E1", None, "special/admin", 1),
            ("USH0002007H1", None, "special/admin", 1),
            ("USPP012345P2", None, "special/admin", 1),
            ("USD012345S1", None, "special/admin", 1),
            # --- Unknown/Non-US Formats ---
            ("WO2015123456A1", None, "unknown_format", 1),
            ("EP1234567A1", None, "unknown_format", 1),
            ("garbage-string", None, "unknown_format", 1),
            ("", None, "unknown_format", 1),
        ],
    )
    def test_get_year_from_patent_id(self, patent_id, expected_year, expected_category, expected_cat_count):
        """
        tests the _get_year_from_patent_id helper with various patent formats.
        """
        adapter = PaRoutesAdapter()
        result = adapter._get_year_from_patent_id(patent_id)

        assert result == expected_year

        if expected_category:
            assert adapter.unparsed_categories[expected_category] == expected_cat_count
            assert sum(adapter.unparsed_categories.values()) == expected_cat_count
        else:
            assert not adapter.unparsed_categories

        assert not adapter.year_counts  # this should only be touched by the main adapt loop


class TestPaRoutesAdapterCycleDetection:
    """tests for cycle detection in paroutes adapter."""

    @pytest.fixture
    def adapter(self) -> PaRoutesAdapter:
        return PaRoutesAdapter()

    def test_simple_cycle_detection_a_to_b_to_a(self, adapter):
        """
        tests that a simple cycle (A -> B -> A) is detected and raises AdapterLogicError.

        Structure:
        Target (A)
          -> Reaction
            -> Reactant (B)
              -> Reaction
                -> Reactant (A)  [CYCLE!]
        """
        # Create cyclic structure where molecule A appears twice
        smiles_a = "CCO"  # ethanol
        smiles_b = "CC(C)O"  # isopropanol

        raw_route_with_cycle = {
            "type": "mol",
            "smiles": smiles_a,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": smiles_a,
                    "metadata": {"ID": "US20150051201A1;0516;1654836"},
                    "children": [
                        {
                            "type": "mol",
                            "smiles": smiles_b,
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": smiles_b,
                                    "metadata": {
                                        "ID": "US20150051201A1;0517;1654837",
                                        "rsmi": f"{smiles_a}>>{smiles_b}",
                                    },
                                    "children": [
                                        {
                                            "type": "mol",
                                            "smiles": smiles_a,  # Same as target - creates cycle!
                                            "in_stock": False,
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        target_input = TargetInput(id="cycle-test-1", smiles=canonicalize_smiles(smiles_a))

        # Should raise AdapterLogicError due to cycle
        routes = list(adapter.cast(raw_route_with_cycle, target_input))
        assert len(routes) == 0  # Cycle should cause transformation to fail

    def test_self_loop_cycle_detection(self, adapter):
        """
        tests that a self-loop (A -> A) is detected and raises AdapterLogicError.

        Structure:
        Target (A)
          -> Reaction
            -> Reactant (A)  [CYCLE!]
        """
        smiles_a = "CCO"

        raw_route_with_self_loop = {
            "type": "mol",
            "smiles": smiles_a,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": smiles_a,
                    "metadata": {"ID": "US20150051201A1;0516;1654836", "rsmi": f"{smiles_a}>>{smiles_a}"},
                    "children": [
                        {
                            "type": "mol",
                            "smiles": smiles_a,  # Same as target - self loop!
                            "in_stock": False,
                            "children": [],
                        }
                    ],
                }
            ],
        }

        target_input = TargetInput(id="self-loop-test", smiles=canonicalize_smiles(smiles_a))

        # Should raise AdapterLogicError due to self-loop
        routes = list(adapter.cast(raw_route_with_self_loop, target_input))
        assert len(routes) == 0

    def test_deep_cycle_detection_a_to_b_to_c_to_b(self, adapter):
        """
        tests that a deeper cycle (A -> B -> C -> B) is detected.

        Structure:
        Target (A)
          -> Reaction
            -> Reactant (B)
              -> Reaction
                -> Reactant (C)
                  -> Reaction
                    -> Reactant (B)  [CYCLE!]
        """
        smiles_a = "CCO"  # ethanol
        smiles_b = "CC(C)O"  # isopropanol
        smiles_c = "CCCO"  # propanol

        raw_route_with_deep_cycle = {
            "type": "mol",
            "smiles": smiles_a,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": smiles_a,
                    "metadata": {"ID": "US20150051201A1;0516;1654836"},
                    "children": [
                        {
                            "type": "mol",
                            "smiles": smiles_b,
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": smiles_b,
                                    "metadata": {"ID": "US20150051201A1;0517;1654837"},
                                    "children": [
                                        {
                                            "type": "mol",
                                            "smiles": smiles_c,
                                            "in_stock": False,
                                            "children": [
                                                {
                                                    "type": "reaction",
                                                    "smiles": smiles_c,
                                                    "metadata": {"ID": "US20150051201A1;0518;1654838"},
                                                    "children": [
                                                        {
                                                            "type": "mol",
                                                            "smiles": smiles_b,  # B appears again - cycle!
                                                            "in_stock": False,
                                                            "children": [],
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        target_input = TargetInput(id="deep-cycle-test", smiles=canonicalize_smiles(smiles_a))

        # Should raise AdapterLogicError due to deep cycle
        routes = list(adapter.cast(raw_route_with_deep_cycle, target_input))
        assert len(routes) == 0

    def test_valid_acyclic_route_no_false_positives(self, adapter, raw_paroutes_data):
        """
        tests that valid acyclic routes still work correctly (no regression).
        This ensures cycle detection doesn't break existing functionality.
        """
        target_id = "paroutes-ex-1"
        raw_route = raw_paroutes_data[target_id]
        target_input = TargetInput(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        routes = list(adapter.cast(raw_route, target_input))

        # Should successfully process the route
        assert len(routes) == 1
        assert routes[0].target.smiles == target_input.smiles
        assert routes[0].length == 2

    def test_branching_route_with_same_leaf_molecule(self, adapter):
        """
        tests that having the same leaf molecule in different branches is NOT a cycle.

        Structure:
        Target (A)
          -> Reaction
            -> Reactant (B)
              -> Reaction
                -> Reactant (C) [leaf]
            -> Reactant (D)
              -> Reaction
                -> Reactant (C) [same leaf, but different path - OK!]
        """
        smiles_a = "CCO"  # target
        smiles_b = "CC(C)O"  # reactant 1
        smiles_c = "C"  # shared leaf (methane)
        smiles_d = "CCCO"  # reactant 2

        raw_route_branching = {
            "type": "mol",
            "smiles": smiles_a,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": smiles_a,
                    "metadata": {"ID": "US20150051201A1;0516;1654836"},
                    "children": [
                        {
                            "type": "mol",
                            "smiles": smiles_b,
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": smiles_b,
                                    "metadata": {"ID": "US20150051201A1;0517;1654837"},
                                    "children": [
                                        {
                                            "type": "mol",
                                            "smiles": smiles_c,
                                            "in_stock": True,
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "type": "mol",
                            "smiles": smiles_d,
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": smiles_d,
                                    "metadata": {"ID": "US20150051201A1;0518;1654838"},
                                    "children": [
                                        {
                                            "type": "mol",
                                            "smiles": smiles_c,  # Same leaf as in other branch
                                            "in_stock": True,
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        target_input = TargetInput(id="branching-test", smiles=canonicalize_smiles(smiles_a))

        # Should successfully process - shared leaves are OK
        routes = list(adapter.cast(raw_route_branching, target_input))
        assert len(routes) == 1
        assert routes[0].length == 2
