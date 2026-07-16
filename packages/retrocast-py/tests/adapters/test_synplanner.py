from __future__ import annotations

import logging

import pytest

from retrocast.adapters.synplanner import SynPlannerAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.models.task import Target
from retrocast.typing import SmilesStr
from tests.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str) -> Target:
    canon_smiles = canonicalize_smiles(smiles, remove_mapping=True)
    return Target(id="synplanner-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_synplanner_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "[CH3:1].[CH3:2]>>[CH3:1][CH2:2]O",
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True},
                    {"type": "mol", "smiles": "CC", "in_stock": True},
                ],
            }
        ],
    }


@pytest.fixture
def raw_synplanner_payload(raw_synplanner_route) -> list[dict]:
    return [raw_synplanner_route, {"type": "mol", "smiles": "CCC", "in_stock": True}]


@pytest.fixture
def raw_synplanner_invalid_leaf_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "C.C>>CCO",
                "children": [{"type": "mol", "smiles": "C"}, {"type": "mol", "smiles": "not-smiles"}],
            }
        ],
    }


@pytest.fixture
def raw_synplanner_null_reactant_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CC(C)(C(=O)O)Br.C>>CCO",
                "children": [
                    None,
                    {"type": "mol", "smiles": "CC(C)(C(=O)O)Br", "in_stock": True},
                ],
            }
        ],
    }


# SECTION: Shared Contract Suite


class TestSynPlannerAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_synplanner_payload, raw_synplanner_route, raw_synplanner_invalid_leaf_route
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=SynPlannerAdapter(),
            extraction=RawExtractionContractCase(
                raw_synplanner_payload, {"type": "mol"}, "synplanner-run", 2, ["synplanner-run", "synplanner-run"], 1
            ),
            casting=CastContractCase(
                raw_synplanner_route,
                {"type": "mol"},
                target_for("CCO"),
                Target(id="synplanner-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(raw_synplanner_invalid_leaf_route, ["C"]),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_synplanner_preserves_reaction_smiles_annotation(raw_synplanner_route) -> None:
    reaction = SynPlannerAdapter().cast(raw_synplanner_route, target=target_for("CCO")).reaction_at("rc:r:/").value
    assert reaction.mapped_reaction_smiles == "[CH3:1].[CH3:2]>>[CH3:1][CH2:2]O"
    assert reaction.annotations == {"source_smiles": "[CH3:1].[CH3:2]>>[CH3:1][CH2:2]O"}


@pytest.mark.contract
def test_synplanner_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(SynPlannerAdapter().iter_raw_routes({"not": "a list"}))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_synplanner_valid_route_list_yields_all_routes(raw_synplanner_payload) -> None:
    entries = list(SynPlannerAdapter().iter_raw_routes(raw_synplanner_payload, source_key="synplanner-run"))

    assert len(entries) == 2
    assert [entry.source_key for entry in entries] == ["synplanner-run", "synplanner-run"]
    assert [entry.source_order for entry in entries] == [1, 2]


@pytest.mark.contract
def test_synplanner_skips_route_with_null_reaction_child(
    raw_synplanner_route, raw_synplanner_null_reactant_route, caplog
) -> None:
    caplog.set_level(logging.WARNING, logger="retrocast.adapters.synplanner")

    entries = list(
        SynPlannerAdapter().iter_raw_routes(
            [raw_synplanner_null_reactant_route, raw_synplanner_route],
            source_key="Apararenone",
        )
    )

    assert len(entries) == 1
    assert entries[0].payload.smiles == "CCO"
    assert entries[0].source_order == 2
    assert "skipping invalid synplanner route: target=Apararenone source_index=0 source_order=1" in caplog.text
    assert "children.0" in caplog.text
    assert "skipped 1 invalid synplanner route(s) for target Apararenone; valid_routes=1" in caplog.text


@pytest.mark.contract
def test_synplanner_all_invalid_routes_raise_explicit_failure(raw_synplanner_null_reactant_route) -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(SynPlannerAdapter().iter_raw_routes([raw_synplanner_null_reactant_route], source_key="Apararenone"))

    assert exc_info.value.code == "adapter.schema_invalid"
    assert "no valid routes" in str(exc_info.value)
    assert exc_info.value.context["adapter"] == "synplanner"
    assert exc_info.value.context["target_id"] == "Apararenone"
    assert exc_info.value.context["skipped_routes"] == 1
    assert exc_info.value.context["first_invalid_source_index"] == 0
    assert exc_info.value.context["first_invalid_source_order"] == 1
    assert "children.0" in exc_info.value.context["first_validation_error"]


@pytest.mark.contract
def test_synplanner_prune_does_not_drop_null_reaction_child(raw_synplanner_null_reactant_route) -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        SynPlannerAdapter().cast(raw_synplanner_null_reactant_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_synplanner_rejects_cycles_after_canonicalization() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [{"type": "reaction", "smiles": "CCO", "children": [{"type": "mol", "smiles": "OCC"}]}],
    }
    with pytest.raises(AdapterLogicError) as exc_info:
        SynPlannerAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_synplanner_allows_duplicate_leaf_molecules() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [{"type": "mol", "smiles": "C"}, {"type": "mol", "smiles": "C"}],
            }
        ],
    }
    route = SynPlannerAdapter().cast(raw_route, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_synplanner_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "C>>CCO",
                "children": [{"type": "mol", "smiles": "not-smiles"}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        SynPlannerAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"
