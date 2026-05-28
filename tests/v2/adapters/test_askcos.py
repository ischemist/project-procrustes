from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, UnsupportedAdapterFeatureError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.askcos import AskcosAdapter
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str, target_id: str = "askcos-target") -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


def askcos_output(pathways: list[list[dict[str, str]]] | None = None) -> dict[str, Any]:
    pathway = [
        {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
        {"source": "uuid-rxn", "target": "uuid-ethanol"},
        {"source": "uuid-rxn", "target": "uuid-acid"},
    ]
    return {
        "results": {
            "stats": {
                "total_iterations": 10,
                "total_chemicals": 3,
                "total_reactions": 1,
                "total_templates": 1,
                "total_paths": 2,
            },
            "uds": {
                "node_dict": {
                    "CCOC(C)=O": {"smiles": "CCOC(C)=O", "id": "chem-root", "type": "chemical", "terminal": False},
                    "CCO": {"smiles": "CCO", "id": "chem-ethanol", "type": "chemical", "terminal": True},
                    "CC(=O)O": {"smiles": "CC(=O)O", "id": "chem-acid", "type": "chemical", "terminal": True},
                    "CC(=O)O.CCO>>CCOC(C)=O": {
                        "smiles": "CC(=O)O.CCO>>CCOC(C)=O",
                        "id": "rxn-1",
                        "type": "reaction",
                        "reaction_properties": {"mapped_smiles": "CC(=O)O.CCO>>CCOC(C)=O"},
                        "model_metadata": [{"source": {"template": {"reaction_smarts": "esterification"}}}],
                    },
                },
                "uuid2smiles": {
                    "00000000-0000-0000-0000-000000000000": "CCOC(C)=O",
                    "uuid-rxn": "CC(=O)O.CCO>>CCOC(C)=O",
                    "uuid-ethanol": "CCO",
                    "uuid-acid": "CC(=O)O",
                },
                "pathways": pathways if pathways is not None else [pathway, pathway],
            },
        }
    }


def invalid_leaf_output() -> dict[str, Any]:
    raw_payload = askcos_output(pathways=None)
    raw_payload["results"]["uds"]["node_dict"]["not-smiles"] = {
        "smiles": "not-smiles",
        "id": "bad-leaf",
        "type": "chemical",
        "terminal": True,
    }
    raw_payload["results"]["uds"]["uuid2smiles"]["uuid-bad"] = "not-smiles"
    raw_payload["results"]["uds"]["pathways"] = [
        [
            {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
            {"source": "uuid-rxn", "target": "uuid-ethanol"},
            {"source": "uuid-rxn", "target": "uuid-bad"},
        ]
    ]
    return raw_payload


@pytest.fixture
def raw_askcos_output() -> dict[str, Any]:
    return askcos_output()


@pytest.fixture
def askcos_route_payload(raw_askcos_output):
    return next(AskcosAdapter().iter_raw_routes(raw_askcos_output)).payload


@pytest.fixture
def askcos_invalid_leaf_payload():
    return next(AskcosAdapter().iter_raw_routes(invalid_leaf_output())).payload


# SECTION: Shared Contract Suite


class TestAskcosAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self,
        raw_askcos_output,
        askcos_route_payload,
        askcos_invalid_leaf_payload,
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=AskcosAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_askcos_output,
                malformed_payload={"results": {}},
                source_key="askcos-run-1",
                expected_entry_count=2,
                expected_source_keys=["askcos-run-1", "askcos-run-1"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=askcos_route_payload,
                malformed_raw_route={"not": "a pathway payload"},
                target=target_for("CCOC(C)=O"),
                mismatched_target=Target(id="askcos-target", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO")),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=askcos_invalid_leaf_payload,
                expected_pruned_root_reactants=["CCO"],
            ),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_askcos_cast_preserves_run_annotations(askcos_route_payload) -> None:
    route = AskcosAdapter().cast(askcos_route_payload, target=target_for("CCOC(C)=O"))

    assert route.annotations == {
        "total_iterations": 10,
        "total_chemicals": 3,
        "total_reactions": 1,
        "total_templates": 1,
        "total_paths": 2,
    }


@pytest.mark.contract
def test_askcos_reaction_fields_and_annotations(askcos_route_payload) -> None:
    route = AskcosAdapter().cast(askcos_route_payload, target=target_for("CCOC(C)=O"))
    reaction = route.reaction_at("rc:r:/").value

    assert reaction.mapped_reaction_smiles == "CC(=O)O.CCO>>CCOC(C)=O"
    assert reaction.template == "esterification"
    assert reaction.annotations == {"source_id": "rxn-1"}


@pytest.mark.contract
def test_askcos_use_full_graph_raises_typed_unsupported_feature(raw_askcos_output) -> None:
    with pytest.raises(UnsupportedAdapterFeatureError) as exc_info:
        list(AskcosAdapter(use_full_graph=True).iter_raw_routes(raw_askcos_output))

    assert exc_info.value.code == "adapter.unsupported_feature"
    assert exc_info.value.context == {"adapter": "askcos", "feature": "full_graph"}


@pytest.mark.contract
def test_askcos_rejects_missing_root_uuid(raw_askcos_output) -> None:
    raw_payload = deepcopy(raw_askcos_output)
    raw_payload["results"]["uds"]["uuid2smiles"].pop("00000000-0000-0000-0000-000000000000")
    raw_route = next(AskcosAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        AskcosAdapter().cast(raw_route, target=target_for("CCOC(C)=O"))

    assert exc_info.value.code == "adapter.node_missing"


@pytest.mark.contract
def test_askcos_rejects_missing_reaction_uuid(raw_askcos_output) -> None:
    raw_payload = deepcopy(raw_askcos_output)
    raw_payload["results"]["uds"]["pathways"][0][0]["target"] = "missing-rxn"
    raw_route = next(AskcosAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        AskcosAdapter().cast(raw_route, target=target_for("CCOC(C)=O"))

    assert exc_info.value.code == "adapter.node_missing"


@pytest.mark.contract
def test_askcos_rejects_cycles(raw_askcos_output) -> None:
    raw_payload = deepcopy(raw_askcos_output)
    raw_payload["results"]["uds"]["pathways"][0] = [
        {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
        {"source": "uuid-rxn", "target": "00000000-0000-0000-0000-000000000000"},
    ]
    raw_route = next(AskcosAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        AskcosAdapter().cast(raw_route, target=target_for("CCOC(C)=O"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_askcos_allows_duplicate_leaf_molecules(raw_askcos_output) -> None:
    raw_payload = deepcopy(raw_askcos_output)
    raw_payload["results"]["uds"]["uuid2smiles"]["uuid-ethanol-2"] = "CCO"
    raw_payload["results"]["uds"]["pathways"][0] = [
        {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
        {"source": "uuid-rxn", "target": "uuid-ethanol"},
        {"source": "uuid-rxn", "target": "uuid-ethanol-2"},
    ]
    raw_route = next(AskcosAdapter().iter_raw_routes(raw_payload)).payload

    route = AskcosAdapter().cast(raw_route, target=target_for("CCOC(C)=O"))

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["CCO", "CCO"]


@pytest.mark.contract
def test_askcos_rejects_empty_reaction_in_strict_mode(raw_askcos_output) -> None:
    raw_payload = deepcopy(raw_askcos_output)
    raw_payload["results"]["uds"]["pathways"][0] = [
        {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"}
    ]
    raw_route = next(AskcosAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        AskcosAdapter().cast(raw_route, target=target_for("CCOC(C)=O"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_askcos_iter_raw_routes_rejects_non_mapping_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(AskcosAdapter().iter_raw_routes(["not", "a", "payload"], source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"
