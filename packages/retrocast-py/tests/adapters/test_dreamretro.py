from __future__ import annotations

import pytest

from retrocast.adapters.dreamretro import DreamRetroErAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, InvalidSmilesError
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


def target_for(smiles: str, target_id: str = "dreamretro-target") -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_dreamretro_payload() -> dict:
    return {
        "succ": True,
        "routes": "CCO>0.9>CC=O.[H][H]",
        "expand_model_call": 4,
        "value_model_call": 2,
        "reaction_nodes_lens": [1],
        "mol_nodes_lens": [3],
    }


@pytest.fixture
def dreamretro_route_payload(raw_dreamretro_payload):
    return next(DreamRetroErAdapter().iter_raw_routes(raw_dreamretro_payload, source_key="dreamretro-run-1")).payload


@pytest.fixture
def dreamretro_invalid_leaf_payload():
    raw_payload = {"succ": True, "routes": "CCO>0.9>C.not-smiles"}
    return next(DreamRetroErAdapter().iter_raw_routes(raw_payload)).payload


# SECTION: Shared Contract Suite


class TestDreamRetroAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self,
        raw_dreamretro_payload,
        dreamretro_route_payload,
        dreamretro_invalid_leaf_payload,
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=DreamRetroErAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_dreamretro_payload,
                malformed_payload={"succ": True, "routes": 123},
                source_key="dreamretro-run-1",
                expected_entry_count=1,
                expected_source_keys=["dreamretro-run-1"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=dreamretro_route_payload,
                malformed_raw_route={"not": "a route payload"},
                target=target_for("CCO"),
                mismatched_target=Target(
                    id="dreamretro-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")
                ),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=dreamretro_invalid_leaf_payload,
                expected_pruned_root_reactants=["C"],
            ),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_dreamretro_preserves_run_annotations(dreamretro_route_payload) -> None:
    route = DreamRetroErAdapter().cast(dreamretro_route_payload, target=target_for("CCO"))

    assert route.annotations == {
        "expand_model_call": 4,
        "value_model_call": 2,
        "reaction_nodes_lens": [1],
        "mol_nodes_lens": [3],
    }


@pytest.mark.contract
def test_dreamretro_payload_annotations_are_immutable_and_private(raw_dreamretro_payload) -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes(raw_dreamretro_payload)).payload
    raw_dreamretro_payload["expand_model_call"] = 999

    with pytest.raises(AttributeError):
        raw_route.annotations = ()

    route = DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"))

    assert route.annotations["expand_model_call"] == 4


@pytest.mark.contract
def test_dreamretro_payloads_use_value_equality() -> None:
    payloads = [
        next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO", "expand_model_call": 4})).payload,
        next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO", "expand_model_call": 4})).payload,
    ]

    assert payloads[0] == payloads[1]


@pytest.mark.contract
def test_dreamretro_iter_raw_routes_skips_unsuccessful_runs() -> None:
    entries = list(DreamRetroErAdapter().iter_raw_routes({"succ": False, "routes": "CCO>0.9>C"}))

    assert entries == []


@pytest.mark.contract
def test_dreamretro_accepts_purchasable_target_route() -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO"})).payload

    route = DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"))

    assert route.target.product_of is None


@pytest.mark.contract
def test_dreamretro_rejects_empty_route_string() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        DreamRetroErAdapter()._parse_route_string("")

    assert exc_info.value.code == "adapter.route_string_empty"


@pytest.mark.contract
def test_dreamretro_rejects_malformed_route_step() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        DreamRetroErAdapter()._parse_route_string("CCO>CC=O")

    assert exc_info.value.code == "adapter.route_string_invalid"


@pytest.mark.contract
def test_dreamretro_strict_rejects_invalid_intermediate_product_smiles() -> None:
    raw_route = next(
        DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C.not-smiles|not-smiles>0.8>C"})
    ).payload

    with pytest.raises(InvalidSmilesError) as exc_info:
        DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"), mode="strict")

    assert exc_info.value.code == "chem.invalid_smiles"


@pytest.mark.contract
def test_dreamretro_prune_skips_invalid_intermediate_product_smiles() -> None:
    raw_route = next(
        DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C.not-smiles|not-smiles>0.8>C"})
    ).payload

    route = DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C"]


@pytest.mark.contract
def test_dreamretro_rejects_cycles_after_smiles_canonicalization() -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C|C>0.8>OCC"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_dreamretro_allows_duplicate_leaf_molecules() -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>C.C"})).payload

    route = DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"))

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_dreamretro_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>not-smiles"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_dreamretro_rejects_empty_reaction_in_strict_mode() -> None:
    raw_route = next(DreamRetroErAdapter().iter_raw_routes({"succ": True, "routes": "CCO>0.9>"})).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        DreamRetroErAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_dreamretro_iter_raw_routes_rejects_non_mapping_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(DreamRetroErAdapter().iter_raw_routes(["not", "a", "payload"], source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"
