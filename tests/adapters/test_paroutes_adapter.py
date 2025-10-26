from copy import deepcopy

import pytest

from tests.adapters.test_base_adapter import BaseAdapterTest
from ursa.adapters.paroutes_adapter import PaRoutesAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo


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
    def target_info(self, raw_paroutes_data):
        smiles = raw_paroutes_data["paroutes-ex-1"]["smiles"]
        return TargetInfo(id="paroutes-ex-1", smiles=canonicalize_smiles(smiles))

    @pytest.fixture
    def mismatched_target_info(self, raw_paroutes_data):
        return TargetInfo(id="paroutes-ex-1", smiles="CCO")  # clearly not the same molecule


@pytest.mark.integration
class TestPaRoutesAdapterIntegration:
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
        target_info = TargetInfo(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # both reaction steps in this example are from patent 'us20150051201a1'.
        trees = list(adapter.adapt(raw_route, target_info))

        assert len(trees) == 1
        tree = trees[0]
        assert tree.target.id == target_id
        assert tree.retrosynthetic_tree.smiles == target_info.smiles
        assert not tree.retrosynthetic_tree.is_starting_material
        # check that it has some depth
        reaction = tree.retrosynthetic_tree.reactions[0]
        assert len(reaction.reactants) == 2
        # check one level deeper
        intermediate_mol = next(r for r in reaction.reactants if not r.is_starting_material)
        assert len(intermediate_mol.reactions) == 1

    def test_rejects_mixed_patent_route(self, adapter, raw_paroutes_data):
        """
        tests that a route is REJECTED if its reaction steps come from
        different patents. this is the key custom logic for this adapter.
        """
        target_id = "paroutes-ex-1"
        # use deepcopy to avoid state leakage between tests
        raw_route = deepcopy(raw_paroutes_data[target_id])
        target_info = TargetInfo(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # let's mutate the data to create the failure condition.
        # the first reaction id is 'us20150051201a1;0516;1654836'
        # we'll change the second one.
        # path: children[0] (reaction) -> children[1] (mol) -> children[0] (reaction)
        inner_reaction = raw_route["children"][0]["children"][1]["children"][0]
        inner_reaction["metadata"]["ID"] = "SOME-OTHER-PATENT;1234;56789"

        # the adapter should now see two different patent ids and yield nothing.
        trees = list(adapter.adapt(raw_route, target_info))

        assert len(trees) == 0

    def test_adapt_second_example_route(self, adapter, raw_paroutes_data):
        """
        tests adaptation on the second example to ensure robustness.
        """
        target_id = "paroutes-ex-2"
        raw_route = raw_paroutes_data[target_id]
        target_info = TargetInfo(id=target_id, smiles=canonicalize_smiles(raw_route["smiles"]))

        # all reaction steps in this example are from patent 'us08242133b2'.
        trees = list(adapter.adapt(raw_route, target_info))

        assert len(trees) == 1
        tree = trees[0]
        assert tree.target.id == target_id
        assert tree.retrosynthetic_tree.smiles == target_info.smiles

        # just check the first reaction's children
        reaction1 = tree.retrosynthetic_tree.reactions[0]
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
