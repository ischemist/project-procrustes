from __future__ import annotations

import pytest

from retrocast.adapters.llm_raw_answers_adapter import LlmRawAnswersAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

# canonical SMILES for the three drugs used in the integration fixture
EBASTINE_SMILES = canonicalize_smiles("CC(C)(C)C1=CC=C(C=C1)C(=O)CCCN2CCC(CC2)OC(C3=CC=CC=C3)C4=CC=CC=C4")
SILDENAFIL_SMILES = canonicalize_smiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
TIVOZANIB_SMILES = canonicalize_smiles("COC1=C(OC)C=C2C(OC3=CC(Cl)=C(NC(=O)NC4=NOC(C)=C4)C=C3)=CC=NC2=C1")


def _wrap_step(product_smiles: str, reactant_smiles: list[str]) -> str:
    reactants = "".join(f"<reactant><smiles>{r}</smiles></reactant>" for r in reactant_smiles)
    return f"<synthesis_step><product><smiles>{product_smiles}</smiles></product>{reactants}</synthesis_step>"


class TestLlmRawAnswersAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return LlmRawAnswersAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # acetone (CC(C)=O) from acetic acid + methane
        completion = "<answer>" + _wrap_step("CC(=O)C", ["CC(=O)O", "C"]) + "</answer>"
        return [{"completion": completion}]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # missing required `completion` field — pydantic validation fails
        return [{"wrong_key": "..."}]

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="acetone", smiles="CCC")

    def test_parses_multi_step_route(self, adapter_instance):
        # Route: C -> CC -> CCC
        steps = _wrap_step("CC", ["C"]) + _wrap_step("CCC", ["CC", "C"])
        completion = f"<answer>{steps}</answer>"
        target_input = TargetInput(id="propane", smiles=canonicalize_smiles("CCC"))

        routes = list(adapter_instance.cast([{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CCC"
        assert target.synthesis_step is not None
        reactant_smiles = {r.smiles for r in target.synthesis_step.reactants}
        assert reactant_smiles == {"CC", "C"}

        intermediate = next(r for r in target.synthesis_step.reactants if r.smiles == "CC")
        assert not intermediate.is_leaf
        assert intermediate.synthesis_step is not None
        assert {r.smiles for r in intermediate.synthesis_step.reactants} == {"C"}

    def test_handles_sm_token_format(self, adapter_instance):
        # tokens <sm_C><sm_C><sm_O> should reconstruct to CCO
        completion = (
            "<synthesis_step>"
            "<product><smiles><sm_C><sm_C><sm_O></smiles></product>"
            "<reactant><smiles><sm_C><sm_O></smiles></reactant>"
            "<reactant><smiles>C</smiles></reactant>"
            "</synthesis_step>"
        )
        target_input = TargetInput(id="ethanol", smiles=canonicalize_smiles("CCO"))

        routes = list(adapter_instance.cast([{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CCO"
        assert target.synthesis_step is not None
        assert {r.smiles for r in target.synthesis_step.reactants} == {"CO", "C"}

    def test_strips_think_blocks(self, adapter_instance):
        # think block contains a fake step that must be ignored
        fake = _wrap_step("CCN", ["C", "N"])
        real = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        completion = f"<think>{fake}</think>{real}"
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapter_instance.cast([{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CC(C)=O"
        # ensure the bogus reactants from the think block did NOT leak into the precursor map
        leaves = {leaf.smiles for leaf in target.synthesis_step.reactants}
        assert leaves == {"CC(=O)O", "C"}

    def test_skips_steps_without_product_or_reactants(self, adapter_instance):
        # one good step + one step with no reactants + one step with no product
        good = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        no_reactants = "<synthesis_step><product><smiles>CCN</smiles></product></synthesis_step>"
        no_product = "<synthesis_step><reactant><smiles>C</smiles></reactant></synthesis_step>"
        completion = good + no_reactants + no_product
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapter_instance.cast([{"completion": completion}], target_input))

        assert len(routes) == 1
        assert routes[0].target.smiles == "CC(C)=O"

    def test_yields_one_route_per_completion_with_ranks(self, adapter_instance):
        completion = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        records = [{"completion": completion} for _ in range(5)]
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapter_instance.cast(records, target_input))

        assert len(routes) == 5
        assert [r.rank for r in routes] == [1, 2, 3, 4, 5]

    def test_completion_with_no_steps_is_skipped(self, adapter_instance, caplog):
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))
        routes = list(adapter_instance.cast([{"completion": "no synthesis steps here"}], target_input))
        assert routes == []
        assert "no synthesis steps" in caplog.text

    def test_mixed_valid_and_invalid_completions(self, adapter_instance):
        good = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        bad = _wrap_step("CCN", ["C", "N"])  # wrong product, won't match target
        records = [
            {"completion": good},
            {"completion": bad},
            {"completion": good},
        ]
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapter_instance.cast(records, target_input))

        assert len(routes) == 2
        assert [r.rank for r in routes] == [1, 3]


@pytest.mark.contract
class TestLlmRawAnswersAdapterContract:
    """Contract tests on real LLM completions: verify Route objects are well-formed."""

    @pytest.fixture(scope="class")
    def adapter(self) -> LlmRawAnswersAdapter:
        return LlmRawAnswersAdapter()

    @pytest.fixture(
        scope="class",
        params=[
            ("Ebastine", EBASTINE_SMILES),
            ("Sildenafil", SILDENAFIL_SMILES),
            ("Tivozanib", TIVOZANIB_SMILES),
        ],
        ids=lambda p: p[0],
    )
    def routes(self, adapter, raw_llm_raw_answers_data, request):
        target_id, target_smi = request.param
        payload = raw_llm_raw_answers_data.get(target_smi)
        assert payload is not None, f"target {target_id} not found under canonical smiles key"

        target_input = TargetInput(id=target_id, smiles=target_smi)
        return list(adapter.cast(payload, target_input))

    def test_produces_at_least_one_route(self, routes):
        assert len(routes) >= 1

    def test_all_routes_preserve_strictly_increasing_ranks(self, routes):
        ranks = [route.rank for route in routes]
        assert ranks == sorted(ranks)
        assert len(ranks) == len(set(ranks))
        assert all(rank >= 1 for rank in ranks)

    def test_target_smiles_match(self, routes, request):
        # request.node.callspec.params resolves the parametrized target
        _, expected = request.node.callspec.params["routes"]
        for route in routes:
            assert route.target.smiles == expected

    def test_all_molecules_have_inchikeys(self, routes):
        def check(mol):
            assert mol.inchikey
            if mol.synthesis_step is not None:
                for r in mol.synthesis_step.reactants:
                    check(r)

        for route in routes:
            check(route.target)

    def test_root_is_not_leaf(self, routes):
        for route in routes:
            assert not route.target.is_leaf
            assert route.target.synthesis_step is not None
