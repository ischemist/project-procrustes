from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from retrocast.adapters.paroutes import ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots
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


def target_for(raw_route: dict, target_id: str = "target") -> Target:
    smiles = canonicalize_smiles(raw_route["smiles"])
    return Target(id=target_id, smiles=smiles, inchikey=get_inchi_key(smiles))


def target_for_entry(entry) -> Target:
    smiles = canonicalize_smiles(entry.payload.smiles)
    return Target(id=entry.source_key, smiles=smiles, inchikey=get_inchi_key(smiles))


def load_real_paroutes_payload() -> dict:
    path = Path("tests/testing_data/paroutes.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


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


@pytest.fixture
def raw_paroutes_invalid_leaf_route() -> dict:
    return {
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


@pytest.fixture
def raw_paroutes_payload(raw_paroutes_route) -> dict:
    second_route = deepcopy(raw_paroutes_route)
    second_route["smiles"] = "CCC"
    second_route["children"][0]["smiles"] = "CCC"
    second_route["children"][0]["metadata"]["rsmi"] = "C.CC>>CCC"
    return {"paroutes-ex-1": raw_paroutes_route, "paroutes-ex-2": second_route}


# SECTION: Shared Contract Suite


class TestPaRoutesAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self,
        raw_paroutes_payload,
        raw_paroutes_route,
        raw_paroutes_invalid_leaf_route,
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=PaRoutesAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_paroutes_payload,
                malformed_payload={"smiles": "CCO"},
                source_key="paroutes-fixture",
                expected_entry_count=2,
                expected_source_keys=["paroutes-ex-1", "paroutes-ex-2"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=raw_paroutes_route,
                malformed_raw_route={"smiles": "CCO"},
                target=target_for(raw_paroutes_route, "paroutes-ex-1"),
                mismatched_target=Target(id="paroutes-ex-1", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=raw_paroutes_invalid_leaf_route,
                expected_pruned_root_reactants=["C"],
            ),
        )

    @pytest.mark.contract
    def test_cast_preserves_patent_annotation(self, raw_paroutes_route) -> None:
        adapter = PaRoutesAdapter()

        route = adapter.cast(raw_paroutes_route, target=target_for(raw_paroutes_route, "paroutes-ex-1"))

        assert route.annotations["patent_id"] == "US123"
        assert route.target.product_of is not None
        assert len(route.target.product_of.reactants) == 2


# SECTION: Regression Tests


@pytest.mark.regression
def test_paroutes_iter_raw_routes_accepts_real_fixture_payload() -> None:
    raw_payload = load_real_paroutes_payload()

    entries = list(PaRoutesAdapter().iter_raw_routes(raw_payload))

    assert [entry.source_key for entry in entries] == ["paroutes-ex-1", "paroutes-ex-2"]
    assert [entry.source_order for entry in entries] == [1, 2]


@pytest.mark.regression
def test_paroutes_casts_real_fixture_routes_with_stable_signatures() -> None:
    adapter = PaRoutesAdapter()
    entries = list(adapter.iter_raw_routes(load_real_paroutes_payload()))

    routes = [adapter.cast(entry.payload, target=target_for_entry(entry)) for entry in entries]

    assert [route.signature() for route in routes] == [
        "d79f5952f18331a4c889c073db1aac16a9842c7474900c3cd81aca276184e11e",
        "b83d6ef3188683bb33b26921cb9ba9a0669ad389f695d941a3de6fa420730985",
    ]
    assert [route.annotations["patent_id"] for route in routes] == ["US20150051201A1", "US08242133B2"]
    assert [[reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] for route in routes] == [
        ["CN1CCn2ncc(N)c21", "CNc1nc(Cl)ncc1C(F)(F)F"],
        ["Nc1cc(OC(F)(F)F)ccc1O", "O=C(O)c1ccncc1Cl"],
    ]


# SECTION: Contract Tests


@pytest.mark.contract
def test_paroutes_iter_raw_routes_accepts_single_route_root(raw_paroutes_route) -> None:
    entries = list(PaRoutesAdapter().iter_raw_routes(raw_paroutes_route, source_key="single-target"))

    assert len(entries) == 1
    assert entries[0].source_key == "single-target"
    assert entries[0].source_order == 1


@pytest.mark.contract
def test_paroutes_iter_raw_routes_rejects_scalar_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(PaRoutesAdapter().iter_raw_routes(42, source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_paroutes_iter_raw_routes_accepts_raw_route_list(raw_paroutes_route) -> None:
    entries = list(PaRoutesAdapter().iter_raw_routes([raw_paroutes_route, {"bad": "route"}], source_key="raw-file"))

    assert len(entries) == 2
    assert [entry.source_key for entry in entries] == ["raw-file", "raw-file"]
    assert [entry.source_row_index for entry in entries] == [1, 2]
    assert [entry.source_order for entry in entries] == [1, 2]


@pytest.mark.contract
def test_paroutes_iter_raw_routes_rejects_non_string_target_keys(raw_paroutes_route) -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(PaRoutesAdapter().iter_raw_routes({1: raw_paroutes_route}))

    assert exc_info.value.code == "adapter.schema_invalid"


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
def test_paroutes_rejects_mixed_patents(raw_paroutes_route) -> None:
    adapter = PaRoutesAdapter()
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["children"][1]["children"][0]["metadata"]["ID"] = "OTHER;1234"

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"))
    assert exc_info.value.code == "adapter.multiple_patents"


@pytest.mark.contract
def test_paroutes_patent_scan_uses_only_selected_reaction(raw_paroutes_route) -> None:
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"].append(
        {
            "type": "reaction",
            "smiles": raw_route["smiles"],
            "metadata": {"ID": "OTHER;1", "rsmi": "C>>CCO"},
            "children": [{"type": "mol", "smiles": "C", "children": []}],
        }
    )

    route = PaRoutesAdapter().cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"))

    assert route.annotations["patent_id"] == "US123"


@pytest.mark.contract
def test_paroutes_rejects_missing_patent_id() -> None:
    raw_route = {"type": "mol", "smiles": "CCO", "in_stock": True, "children": []}

    with pytest.raises(AdapterLogicError) as exc_info:
        PaRoutesAdapter().cast(raw_route, target=target_for(raw_route, "missing-patent"))

    assert exc_info.value.code == "adapter.patent_id_missing"


@pytest.mark.contract
def test_paroutes_rejects_empty_patent_id() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {"type": "reaction", "smiles": "CCO", "metadata": {"ID": " ;1", "rsmi": "C>>CCO"}, "children": []}
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        PaRoutesAdapter().cast(raw_route, target=target_for(raw_route, "empty-patent"))

    assert exc_info.value.code == "adapter.patent_id_missing"


@pytest.mark.contract
def test_paroutes_rejects_molecule_child_under_molecule() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [{"type": "mol", "smiles": "C", "children": []}],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        PaRoutesAdapter().cast(raw_route, target=target_for(raw_route, "bad-topology"))

    assert exc_info.value.code == "adapter.node_type_invalid"


@pytest.mark.contract
def test_paroutes_rejects_cycles_before_patent_checks() -> None:
    adapter = PaRoutesAdapter()
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "CC>>CCO"},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "CC",
                        "children": [
                            {
                                "type": "reaction",
                                "smiles": "CC",
                                "metadata": {"ID": "US123;2", "rsmi": "CCO>>CC"},
                                "children": [{"type": "mol", "smiles": "CCO", "children": []}],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target_for(raw_route, "cycle-target"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_paroutes_rejects_cycles_after_smiles_canonicalization() -> None:
    adapter = PaRoutesAdapter()
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "OCC>>CCO"},
                "children": [{"type": "mol", "smiles": "OCC", "children": []}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target_for(raw_route, "cycle-target"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_paroutes_rejects_empty_reaction_in_strict_mode() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {"type": "reaction", "smiles": "CCO", "metadata": {"ID": "US123;1", "rsmi": ">>CCO"}, "children": []}
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        PaRoutesAdapter().cast(raw_route, target=target_for(raw_route, "empty-reaction"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_paroutes_prune_mode_drops_branch_when_all_reactants_are_invalid() -> None:
    adapter = PaRoutesAdapter()
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "not-smiles>>CCO"},
                "children": [{"type": "mol", "smiles": "not-smiles", "in_stock": True, "children": []}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        adapter.cast(raw_route, target=target_for(raw_route, "paroutes-ex-1"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


# SECTION: Diagnostics Tests


@pytest.mark.contract
def test_analyze_condition_slots_counts_non_fatal_failures(raw_paroutes_route) -> None:
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["metadata"]["rsmi"] = "not-a-valid-rsmi"
    stats = ConditionSlotParseStatistics()

    analyze_condition_slots(raw_route, stats=stats)

    assert stats.malformed_rsmi_count == 1
    assert stats.uncanonicalizable_token_count == 0


@pytest.mark.contract
def test_analyze_condition_slots_counts_uncanonicalizable_tokens(raw_paroutes_route) -> None:
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["metadata"]["rsmi"] = "CCO>not-smiles..O>CCO"
    stats = ConditionSlotParseStatistics()

    analyze_condition_slots(raw_route, stats=stats)

    assert stats.uncanonicalizable_token_count == 1
    assert stats.distinct_uncanonicalizable_token_count == 1
    assert stats.top_uncanonicalizable_tokens == [("not-smiles", 1)]


@pytest.mark.contract
def test_analyze_condition_slots_ignores_missing_rsmi(raw_paroutes_route) -> None:
    raw_route = deepcopy(raw_paroutes_route)
    raw_route["children"][0]["metadata"].pop("rsmi")
    stats = ConditionSlotParseStatistics()

    analyze_condition_slots(raw_route, stats=stats)

    assert stats.malformed_rsmi_count == 0
    assert stats.uncanonicalizable_token_count == 0


@pytest.mark.contract
def test_analyze_condition_slots_ignores_malformed_tree_shapes() -> None:
    stats = ConditionSlotParseStatistics()

    analyze_condition_slots({"children": [{"children": "not-a-list"}, "not-a-node"]}, stats=stats)

    assert stats.malformed_rsmi_count == 0
    assert stats.uncanonicalizable_token_count == 0
