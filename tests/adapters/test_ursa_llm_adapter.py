from __future__ import annotations

from collections import defaultdict

import pytest

from retrocast.adapters.ursa_llm_adapter import UrsaLlmAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterSchemaError
from retrocast.models.chem import Route, TargetInput
from retrocast.workflow.adapt import adapt_route_corpus, adapt_target_routes
from tests.adapters.test_base_adapter import BaseAdapterTest

EBASTINE_SMILES = canonicalize_smiles("CC(C)(C)C1=CC=C(C=C1)C(=O)CCCN2CCC(CC2)OC(C3=CC=CC=C3)C4=CC=CC=C4")
SILDENAFIL_SMILES = canonicalize_smiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
TIVOZANIB_SMILES = canonicalize_smiles("COC1=C(OC)C=C2C(OC3=CC(Cl)=C(NC(=O)NC4=NOC(C)=C4)C=C3)=CC=NC2=C1")


def _wrap_step(product_smiles: str, reactant_smiles: list[str]) -> str:
    reactants = "".join(f"<reactant><smiles>{smiles}</smiles></reactant>" for smiles in reactant_smiles)
    return f"<synthesis_step><product><smiles>{product_smiles}</smiles></product>{reactants}</synthesis_step>"


class TestUrsaLlmAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self) -> UrsaLlmAdapter:
        return UrsaLlmAdapter()

    @pytest.fixture
    def raw_valid_route_data(self) -> list[dict[str, str]]:
        completion = "<answer>" + _wrap_step("CC(=O)C", ["CC(=O)O", "C"]) + "</answer>"
        return [{"completion": completion}]

    @pytest.fixture
    def raw_unsuccessful_run_data(self) -> list[dict[str, str]]:
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self) -> list[dict[str, str]]:
        return [{"wrong_key": "..."}]

    @pytest.fixture
    def target_input(self) -> TargetInput:
        return TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

    @pytest.fixture
    def mismatched_target_input(self) -> TargetInput:
        return TargetInput(id="acetone", smiles="CCC")

    def test_parses_multi_step_route(self, adapter_instance: UrsaLlmAdapter) -> None:
        steps = _wrap_step("CC", ["C"]) + _wrap_step("CCC", ["CC", "C"])
        completion = f"<answer>{steps}</answer>"
        target_input = TargetInput(id="propane", smiles=canonicalize_smiles("CCC"))

        routes = list(adapt_target_routes(adapter_instance, [{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CCC"
        assert target.synthesis_step is not None
        reactant_smiles = {reactant.smiles for reactant in target.synthesis_step.reactants}
        assert reactant_smiles == {"CC", "C"}

        intermediate = next(reactant for reactant in target.synthesis_step.reactants if reactant.smiles == "CC")
        assert not intermediate.is_leaf
        assert intermediate.synthesis_step is not None
        assert {reactant.smiles for reactant in intermediate.synthesis_step.reactants} == {"C"}

    def test_handles_sm_token_format(self, adapter_instance: UrsaLlmAdapter) -> None:
        completion = (
            "<synthesis_step>"
            "<product><smiles><sm_C><sm_C><sm_O></smiles></product>"
            "<reactant><smiles><sm_C><sm_O></smiles></reactant>"
            "<reactant><smiles>C</smiles></reactant>"
            "</synthesis_step>"
        )
        target_input = TargetInput(id="ethanol", smiles=canonicalize_smiles("CCO"))

        routes = list(adapt_target_routes(adapter_instance, [{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CCO"
        assert target.synthesis_step is not None
        assert {reactant.smiles for reactant in target.synthesis_step.reactants} == {"CO", "C"}

    def test_strips_think_blocks(self, adapter_instance: UrsaLlmAdapter) -> None:
        fake = _wrap_step("CCN", ["C", "N"])
        real = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        completion = f"<think>{fake}</think>{real}"
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapt_target_routes(adapter_instance, [{"completion": completion}], target_input))

        assert len(routes) == 1
        target = routes[0].target
        assert target.smiles == "CC(C)=O"
        assert target.synthesis_step is not None
        leaves = {leaf.smiles for leaf in target.synthesis_step.reactants}
        assert leaves == {"CC(=O)O", "C"}

    def test_skips_steps_without_product_or_reactants(self, adapter_instance: UrsaLlmAdapter) -> None:
        good = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        no_reactants = "<synthesis_step><product><smiles>CCN</smiles></product></synthesis_step>"
        no_product = "<synthesis_step><reactant><smiles>C</smiles></reactant></synthesis_step>"
        completion = good + no_reactants + no_product
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapt_target_routes(adapter_instance, [{"completion": completion}], target_input))

        assert len(routes) == 1
        assert routes[0].target.smiles == "CC(C)=O"

    def test_iter_raw_entries_uses_source_target_metadata_without_benchmark_context(
        self,
        adapter_instance: UrsaLlmAdapter,
    ) -> None:
        completion = _wrap_step("c1ccccc1", ["C", "CC"])
        entries = list(
            adapter_instance.iter_raw_entries([{"meta": {"product_smiles": "C1=CC=CC=C1"}, "completion": completion}])
        )

        assert len(entries) == 1
        assert entries[0].expected_target_smiles == canonicalize_smiles("C1=CC=CC=C1")
        assert entries[0].payload == completion

    def test_iter_raw_entries_requires_source_target_metadata_without_expected_target(
        self,
        adapter_instance: UrsaLlmAdapter,
    ) -> None:
        with pytest.raises(AdapterSchemaError) as exc_info:
            list(adapter_instance.iter_raw_entries([{"completion": "route-1"}]))

        assert exc_info.value.code == "adapter.schema_invalid"

    def test_yields_one_route_per_completion(self, adapter_instance: UrsaLlmAdapter) -> None:
        completion = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        records = [{"completion": completion} for _ in range(5)]
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapt_target_routes(adapter_instance, records, target_input))

        assert len(routes) == 5
        assert all(route.target.smiles == target_input.smiles for route in routes)

    def test_completion_with_no_steps_is_skipped(self, adapter_instance: UrsaLlmAdapter, caplog) -> None:
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapt_target_routes(adapter_instance, [{"completion": "no synthesis steps here"}], target_input))

        assert routes == []
        assert "no synthesis steps" in caplog.text

    def test_mixed_valid_and_invalid_completions(self, adapter_instance: UrsaLlmAdapter) -> None:
        good = _wrap_step("CC(=O)C", ["CC(=O)O", "C"])
        bad = _wrap_step("CCN", ["C", "N"])
        records = [
            {"completion": good},
            {"completion": bad},
            {"completion": good},
        ]
        target_input = TargetInput(id="acetone", smiles=canonicalize_smiles("CC(C)=O"))

        routes = list(adapt_target_routes(adapter_instance, records, target_input))

        assert len(routes) == 2
        assert all(route.target.smiles == target_input.smiles for route in routes)


@pytest.mark.contract
class TestUrsaLlmAdapterContract:
    @pytest.fixture(scope="class")
    def routes_by_target_smiles(self, raw_ursa_llm_data) -> dict[str, list[Route]]:
        route_corpus = adapt_route_corpus(raw_ursa_llm_data, UrsaLlmAdapter())
        grouped_routes: dict[str, list[Route]] = defaultdict(list)
        for route in route_corpus:
            grouped_routes[route.target.smiles].append(route)
        return grouped_routes

    @pytest.fixture(
        scope="class",
        params=[
            ("Ebastine", EBASTINE_SMILES),
            ("Sildenafil", SILDENAFIL_SMILES),
            ("Tivozanib", TIVOZANIB_SMILES),
        ],
        ids=lambda param: param[0],
    )
    def routes(self, routes_by_target_smiles: dict[str, list[Route]], request) -> list[Route]:
        _, target_smiles = request.param
        return routes_by_target_smiles[target_smiles]

    def test_produces_at_least_one_route(self, routes: list[Route]) -> None:
        assert len(routes) >= 1

    def test_target_smiles_match(self, routes: list[Route], request) -> None:
        _, expected = request.node.callspec.params["routes"]
        for route in routes:
            assert route.target.smiles == expected

    def test_all_molecules_have_inchikeys(self, routes: list[Route]) -> None:
        def check(molecule) -> None:
            assert molecule.inchikey
            if molecule.synthesis_step is not None:
                for reactant in molecule.synthesis_step.reactants:
                    check(reactant)

        for route in routes:
            check(route.target)

    def test_root_is_not_leaf(self, routes: list[Route]) -> None:
        for route in routes:
            assert not route.target.is_leaf
            assert route.target.synthesis_step is not None
