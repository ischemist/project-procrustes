from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.dms import DirectMultiStepAdapter, DMSTree
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str, target_id: str = "dms-target") -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_dms_payload() -> list[dict]:
    return [
        {"smiles": "CCO", "children": [{"smiles": "CC=O", "children": []}, {"smiles": "[H][H]", "children": []}]},
        {"smiles": "CCC", "children": []},
    ]


@pytest.fixture
def raw_dms_route(raw_dms_payload):
    return raw_dms_payload[0]


@pytest.fixture
def raw_dms_invalid_leaf_route() -> dict:
    return {"smiles": "CCO", "children": [{"smiles": "C", "children": []}, {"smiles": "not-smiles", "children": []}]}


# SECTION: Shared Contract Suite


class TestDMSAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(self, raw_dms_payload, raw_dms_route, raw_dms_invalid_leaf_route) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=DirectMultiStepAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_dms_payload,
                malformed_payload={"smiles": "CCO"},
                source_key="dms-run-1",
                expected_entry_count=2,
                expected_source_keys=["dms-run-1", "dms-run-1"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=raw_dms_route,
                malformed_raw_route={"children": []},
                target=target_for("CCO"),
                mismatched_target=Target(id="dms-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=raw_dms_invalid_leaf_route,
                expected_pruned_root_reactants=["C"],
            ),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_dms_iter_raw_routes_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(DirectMultiStepAdapter().iter_raw_routes({"not": "a list"}, source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_dms_rejects_cycles_after_canonicalization() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "OCC", "children": []}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        DirectMultiStepAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_dms_allows_duplicate_leaf_molecules() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "C", "children": []}, {"smiles": "C", "children": []}]}

    route = DirectMultiStepAdapter().cast(raw_route, target=target_for("CCO"))

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_dms_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "not-smiles", "children": []}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        DirectMultiStepAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_dms_calculates_route_length() -> None:
    tree = DMSTree.model_validate({"smiles": "CCO", "children": [{"smiles": "CC", "children": [{"smiles": "C"}]}]})

    assert DirectMultiStepAdapter.calculate_route_length(tree) == 2
