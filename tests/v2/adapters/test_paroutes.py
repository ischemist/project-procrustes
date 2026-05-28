from __future__ import annotations

from copy import deepcopy

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.paroutes import ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target


def target_for(raw_route: dict, target_id: str = "target") -> Target:
    smiles = canonicalize_smiles(raw_route["smiles"])
    return Target(id=target_id, smiles=smiles, inchikey=get_inchi_key(smiles))


@pytest.fixture
def raw_paroutes_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "C.CCO>O.C>CCO", "RingBreaker": False},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                    {
                        "type": "mol",
                        "smiles": "CC",
                        "children": [
                            {
                                "type": "reaction",
                                "smiles": "CC",
                                "metadata": {"ID": "US123;2", "rsmi": "C>C>CC"},
                                "children": [{"type": "mol", "smiles": "C", "in_stock": True, "children": []}],
                            }
                        ],
                    },
                ],
            }
        ],
    }


@pytest.mark.contract
def test_paroutes_iter_raw_routes_validates_route_root(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()

    entries = list(adapter.iter_raw_routes(raw_paroutes_route, source_key="paroutes-ex-1"))

    assert len(entries) == 1
    assert entries[0].source_key == "paroutes-ex-1"
    assert entries[0].source_order == 1

    with pytest.raises(AdapterSchemaError) as exc_info:
        list(adapter.iter_raw_routes({"smiles": "CCO"}, source_key="bad"))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_paroutes_cast_emits_route(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()
    raw_route = raw_paroutes_route

    route = adapter.cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"))

    assert isinstance(route, Route)
    assert route.schema_version == "2"
    assert route.annotations["patent_id"] == "US123"
    assert route.target.smiles == canonicalize_smiles(raw_route["smiles"])
    assert route.target.inchikey
    assert route.target.product_of is not None
    assert len(route.target.product_of.reactants) == 2


@pytest.mark.contract
def test_paroutes_reaction_annotations_keep_condition_slot(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()
    raw_route = raw_paroutes_route

    route = adapter.cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"))
    reaction = route.reaction_at("rc:r:/").value
    raw_reaction = raw_route["children"][0]
    condition_slot = raw_reaction["metadata"]["rsmi"].split(">")[1]

    assert reaction.mapped_reaction_smiles == raw_reaction["metadata"]["rsmi"]
    assert reaction.template is None
    assert reaction.annotations["source_id"] == raw_reaction["metadata"]["ID"]
    assert reaction.annotations["ring_breaker"] is False
    assert reaction.annotations["condition_slot"] == condition_slot
    assert reaction.annotations["condition_slot_smiles"] == sorted(
        canonicalize_smiles(token) for token in condition_slot.split(".")
    )
    assert "reaction_hash" not in reaction.annotations
    assert "smiles" not in reaction.annotations


@pytest.mark.contract
def test_paroutes_rejects_mismatched_target(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()
    raw_route = raw_paroutes_route
    target = Target(id="paroutes-ex-1", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC"))

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target)
    assert exc_info.value.code == "adapter.target_mismatch"


@pytest.mark.contract
def test_paroutes_rejects_mixed_patents(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["children"][1]["children"][0]["metadata"]["ID"] = "OTHER;1234"

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"))
    assert exc_info.value.code == "adapter.multiple_patents"


@pytest.mark.contract
def test_paroutes_prune_mode_drops_invalid_branch() -> None:
    adapter = PaRoutesAdapter()
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "C.CCO>>CCO"},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
                    {"type": "mol", "smiles": "not-smiles", "in_stock": True, "children": []},
                ],
            }
        ],
    }

    strict_target = target_for(raw_route, "prune-target")
    with pytest.raises(InvalidSmilesError):
        adapter.cast(raw_route, target=strict_target, mode="strict")

    route = adapter.cast(raw_route, target=strict_target, mode="prune")
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == [SmilesStr("C")]


@pytest.mark.contract
def test_analyze_condition_slots_counts_non_fatal_failures(raw_paroutes_route) -> None:
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["metadata"]["rsmi"] = "not-a-valid-rsmi"
    stats = ConditionSlotParseStatistics()

    analyze_condition_slots(raw_route, stats=stats)

    assert stats.malformed_rsmi_count == 1
    assert stats.uncanonicalizable_token_count == 0
