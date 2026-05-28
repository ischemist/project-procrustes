from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.multistepttl import MultiStepTTLAdapter
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str) -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id="ttl-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_ttl_route() -> dict:
    return {"reactions": [{"product": "CCO", "reactants": ["C", "CC"]}], "metadata": {"score": 0.7}}


@pytest.fixture
def raw_ttl_payload(raw_ttl_route) -> list[dict]:
    return [raw_ttl_route, {"reactions": [], "metadata": {}}]


@pytest.fixture
def raw_ttl_invalid_leaf_route() -> dict:
    return {"reactions": [{"product": "CCO", "reactants": ["C", "not-smiles"]}], "metadata": {}}


# SECTION: Shared Contract Suite


class TestMultiStepTTLAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(self, raw_ttl_payload, raw_ttl_route, raw_ttl_invalid_leaf_route) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=MultiStepTTLAdapter(),
            extraction=RawExtractionContractCase(
                raw_ttl_payload, {"reactions": "bad"}, "ttl-run", 2, ["ttl-run", "ttl-run"], 1
            ),
            casting=CastContractCase(
                raw_ttl_route,
                {"metadata": {}},
                target_for("CCO"),
                Target(id="ttl-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(raw_ttl_invalid_leaf_route, ["C"]),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_multistepttl_preserves_route_metadata(raw_ttl_route) -> None:
    route = MultiStepTTLAdapter().cast(raw_ttl_route, target=target_for("CCO"))
    assert route.annotations == {"score": 0.7}


@pytest.mark.contract
def test_multistepttl_accepts_zero_reaction_route_with_target() -> None:
    route = MultiStepTTLAdapter().cast({"reactions": [], "metadata": {"rank": 1}}, target=target_for("CCO"))
    assert route.target.product_of is None
    assert route.annotations == {"rank": 1}


@pytest.mark.contract
def test_multistepttl_rejects_zero_reaction_route_without_target() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        MultiStepTTLAdapter().cast({"reactions": []})
    assert exc_info.value.code == "adapter.route_transform_failed"


@pytest.mark.contract
def test_multistepttl_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(MultiStepTTLAdapter().iter_raw_routes({"not": "a list"}))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_multistepttl_rejects_cycles_after_canonicalization() -> None:
    raw_route = {"reactions": [{"product": "CCO", "reactants": ["C"]}, {"product": "C", "reactants": ["OCC"]}]}
    with pytest.raises(AdapterLogicError) as exc_info:
        MultiStepTTLAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_multistepttl_allows_duplicate_leaf_molecules() -> None:
    raw_route = {"reactions": [{"product": "CCO", "reactants": ["C", "C"]}]}
    route = MultiStepTTLAdapter().cast(raw_route, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_multistepttl_rejects_empty_reaction_in_strict_mode() -> None:
    raw_route = {"reactions": [{"product": "CCO", "reactants": []}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        MultiStepTTLAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_multistepttl_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {"reactions": [{"product": "CCO", "reactants": ["not-smiles"]}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        MultiStepTTLAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"
