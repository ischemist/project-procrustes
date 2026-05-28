from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.retrostar import RetroStarAdapter
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str, target_id: str = "retrostar-target") -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_retrostar_payload() -> dict:
    return {"succ": True, "routes": "CCO>0.9>CC=O.[H][H]", "route_cost": 1.25}


@pytest.fixture
def retrostar_route_payload(raw_retrostar_payload):
    return next(RetroStarAdapter().iter_raw_routes(raw_retrostar_payload, source_key="retrostar-run-1")).payload


@pytest.fixture
def retrostar_invalid_leaf_payload():
    raw_payload = {"succ": True, "routes": "CCO>0.9>C.not-smiles", "route_cost": 2.0}
    return next(RetroStarAdapter().iter_raw_routes(raw_payload)).payload


# SECTION: Shared Contract Suite


class TestRetroStarAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self,
        raw_retrostar_payload,
        retrostar_route_payload,
        retrostar_invalid_leaf_payload,
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=RetroStarAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_retrostar_payload,
                malformed_payload={"succ": True, "routes": 123},
                source_key="retrostar-run-1",
                expected_entry_count=1,
                expected_source_keys=["retrostar-run-1"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=retrostar_route_payload,
                malformed_raw_route={"not": "a route payload"},
                target=target_for("CCO"),
                mismatched_target=Target(id="retrostar-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=retrostar_invalid_leaf_payload,
                expected_pruned_root_reactants=["C"],
            ),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_retrostar_cast_preserves_route_cost_and_step_score(retrostar_route_payload) -> None:
    route = RetroStarAdapter().cast(retrostar_route_payload, target=target_for("CCO"))

    assert route.annotations["route_cost"] == 1.25
    assert route.reaction_at("rc:r:/").value.annotations == {"step_score": 0.9}


@pytest.mark.contract
def test_retrostar_iter_raw_routes_skips_unsuccessful_runs() -> None:
    entries = list(RetroStarAdapter().iter_raw_routes({"succ": False, "routes": "CCO>0.9>C"}))

    assert entries == []


@pytest.mark.contract
def test_retrostar_accepts_purchasable_target_route() -> None:
    raw_route = next(RetroStarAdapter().iter_raw_routes({"succ": True, "routes": "CCO", "route_cost": 0.0})).payload

    route = RetroStarAdapter().cast(raw_route, target=target_for("CCO"))

    assert route.target.product_of is None
    assert route.annotations["route_cost"] == 0.0


@pytest.mark.contract
def test_retrostar_rejects_empty_route_string() -> None:
    adapter = RetroStarAdapter()

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter._parse_route_string("")

    assert exc_info.value.code == "adapter.route_string_empty"


@pytest.mark.contract
def test_retrostar_rejects_malformed_route_step() -> None:
    adapter = RetroStarAdapter()

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter._parse_route_string("CCO>CC=O")

    assert exc_info.value.code == "adapter.route_string_invalid"


@pytest.mark.contract
def test_retrostar_rejects_malformed_later_route_step() -> None:
    adapter = RetroStarAdapter()

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter._parse_route_string("CCO>0.9>CC=O|CC=O>0.8")

    assert exc_info.value.code == "adapter.route_string_invalid"


@pytest.mark.contract
def test_retrostar_rejects_cycles_after_smiles_canonicalization() -> None:
    raw_route = next(RetroStarAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C|C>0.8>OCC"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroStarAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_retrostar_allows_duplicate_leaf_molecules() -> None:
    raw_route = next(RetroStarAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C.C"})).payload

    route = RetroStarAdapter().cast(raw_route, target=target_for("CCO"))

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_retrostar_rejects_empty_reaction_in_strict_mode() -> None:
    raw_route = next(RetroStarAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroStarAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_retrostar_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = next(RetroStarAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>not-smiles"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroStarAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_retrostar_iter_raw_routes_rejects_non_mapping_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(RetroStarAdapter().iter_raw_routes(["not", "a", "payload"], source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"
