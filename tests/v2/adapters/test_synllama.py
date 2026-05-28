from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.synllama import SynLlamaAdapter
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)


def target_for(smiles: str) -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id="synllama-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_synllama_payload() -> list[dict]:
    return [{"synthesis_string": "C;CC;R1;CCO"}, {"synthesis_string": "CCC"}]


@pytest.fixture
def raw_synllama_route(raw_synllama_payload) -> dict:
    return raw_synllama_payload[0]


@pytest.fixture
def raw_synllama_invalid_leaf_route() -> dict:
    return {"synthesis_string": "C;not-smiles;R1;CCO"}


class TestSynLlamaAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_synllama_payload, raw_synllama_route, raw_synllama_invalid_leaf_route
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=SynLlamaAdapter(),
            extraction=RawExtractionContractCase(
                raw_synllama_payload,
                {"synthesis_string": "CCO"},
                "synllama-run",
                2,
                ["synllama-run", "synllama-run"],
                1,
            ),
            casting=CastContractCase(
                raw_synllama_route,
                {"bad": "route"},
                target_for("CCO"),
                Target(id="synllama-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(raw_synllama_invalid_leaf_route, ["C"]),
        )


@pytest.mark.contract
def test_synllama_accepts_purchasable_target_route() -> None:
    route = SynLlamaAdapter().cast({"synthesis_string": "CCO"}, target=target_for("CCO"))
    assert route.target.product_of is None


@pytest.mark.contract
def test_synllama_rejects_empty_synthesis_string() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        SynLlamaAdapter().cast({"synthesis_string": " ; "}, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.route_string_empty"


@pytest.mark.contract
def test_synllama_rejects_template_without_product() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        SynLlamaAdapter()._parse_synthesis_string("C;R1")
    assert exc_info.value.code == "adapter.route_string_invalid"


@pytest.mark.contract
def test_synllama_rejects_reaction_without_reactants() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        SynLlamaAdapter()._parse_synthesis_string("R1;CCO")
    assert exc_info.value.code == "adapter.route_string_invalid"


@pytest.mark.contract
def test_synllama_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(SynLlamaAdapter().iter_raw_routes({"not": "a list"}))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_synllama_rejects_cycles_after_canonicalization() -> None:
    raw_route = {"synthesis_string": "CCO;R1;C;R2;CCO"}
    with pytest.raises(AdapterLogicError) as exc_info:
        SynLlamaAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_synllama_strict_rejects_invalid_target_product() -> None:
    with pytest.raises(InvalidSmilesError) as exc_info:
        SynLlamaAdapter().cast({"synthesis_string": "C;R1;not-smiles"}, target=target_for("CCO"))
    assert exc_info.value.code == "chem.invalid_smiles"


@pytest.mark.contract
def test_synllama_prune_rejects_invalid_target_product_as_pruned() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        SynLlamaAdapter().cast({"synthesis_string": "C;R1;not-smiles"}, target=target_for("CCO"), mode="prune")
    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_synllama_prune_skips_invalid_intermediate_product() -> None:
    raw_route = {"synthesis_string": "C;R1;not-smiles;CC;R2;CCO"}

    route = SynLlamaAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["CC"]


@pytest.mark.contract
def test_synllama_prune_does_not_carry_over_product_after_failure() -> None:
    raw_route = {"synthesis_string": "C;R1;CC;C;R2;not-smiles;CCC;R3;CCO"}

    route = SynLlamaAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["CCC"]


@pytest.mark.contract
def test_synllama_allows_duplicate_leaf_molecules() -> None:
    route = SynLlamaAdapter().cast({"synthesis_string": "C;C;R1;CCO"}, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]
