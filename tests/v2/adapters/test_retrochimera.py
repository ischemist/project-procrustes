from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.retrochimera import RetroChimeraAdapter
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
    return Target(id="retrochimera-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


def raw_output(route: dict) -> dict:
    return {
        "smiles": "CCO",
        "result": {
            "outputs": [
                {
                    "routes": [route, route],
                    "num_routes": 2,
                    "num_nodes_explored": 5,
                }
            ]
        },
    }


@pytest.fixture
def raw_retrochimera_route() -> dict:
    return {
        "reactions": [{"product": "CCO", "reactants": ["C", "CC"], "probability": 0.8, "metadata": {"model": "x"}}],
        "num_steps": 1,
        "step_probability_min": 0.8,
        "step_probability_product": 0.8,
    }


@pytest.fixture
def raw_retrochimera_payload(raw_retrochimera_route) -> dict:
    return raw_output(raw_retrochimera_route)


@pytest.fixture
def retrochimera_route_payload(raw_retrochimera_payload):
    return next(RetroChimeraAdapter().iter_raw_routes(raw_retrochimera_payload, source_key="retrochimera-run")).payload


@pytest.fixture
def retrochimera_invalid_leaf_payload():
    route = {
        "reactions": [{"product": "CCO", "reactants": ["C", "not-smiles"], "probability": 0.8}],
        "num_steps": 1,
        "step_probability_min": 0.8,
        "step_probability_product": 0.8,
    }
    return next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload


# SECTION: Shared Contract Suite


class TestRetroChimeraAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_retrochimera_payload, retrochimera_route_payload, retrochimera_invalid_leaf_payload
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=RetroChimeraAdapter(),
            extraction=RawExtractionContractCase(
                raw_retrochimera_payload,
                {"result": {}},
                "retrochimera-run",
                2,
                ["retrochimera-run", "retrochimera-run"],
                1,
            ),
            casting=CastContractCase(
                retrochimera_route_payload,
                {"not": "payload"},
                target_for("CCO"),
                Target(id="retrochimera-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(retrochimera_invalid_leaf_payload, ["C"]),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_retrochimera_preserves_output_and_reaction_annotations(retrochimera_route_payload) -> None:
    route = RetroChimeraAdapter().cast(retrochimera_route_payload, target=target_for("CCO"))
    assert route.annotations["num_routes"] == 2
    assert route.annotations["num_nodes_explored"] == 5
    assert route.reaction_at("rc:r:/").value.annotations == {"probability": 0.8, "model": "x"}


@pytest.mark.contract
def test_retrochimera_payload_annotations_are_immutable_and_private(raw_retrochimera_payload) -> None:
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_retrochimera_payload)).payload
    raw_retrochimera_payload["result"]["outputs"][0]["num_nodes_explored"] = 999

    with pytest.raises(AttributeError):
        raw_route.annotations = ()

    route = RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"))

    assert route.annotations["num_nodes_explored"] == 5


@pytest.mark.contract
def test_retrochimera_rejects_model_error() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        list(
            RetroChimeraAdapter().iter_raw_routes(
                {"smiles": "CCO", "result": {"error": {"type": "boom", "message": "bad"}}}
            )
        )
    assert exc_info.value.code == "adapter.route_transform_failed"


@pytest.mark.contract
def test_retrochimera_rejects_missing_outputs() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        list(RetroChimeraAdapter().iter_raw_routes({"smiles": "CCO", "result": {}}))
    assert exc_info.value.code == "adapter.route_transform_failed"


@pytest.mark.contract
def test_retrochimera_rejects_cycles_after_canonicalization() -> None:
    route = {
        "reactions": [
            {"product": "CCO", "reactants": ["C"], "probability": 0.8},
            {"product": "C", "reactants": ["OCC"], "probability": 0.8},
        ],
        "num_steps": 2,
        "step_probability_min": 0.8,
        "step_probability_product": 0.64,
    }
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload
    with pytest.raises(AdapterLogicError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_retrochimera_rejects_empty_reaction_in_strict_mode() -> None:
    route = {
        "reactions": [{"product": "CCO", "reactants": [], "probability": 0.8}],
        "num_steps": 1,
        "step_probability_min": 0.8,
        "step_probability_product": 0.8,
    }
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_retrochimera_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    route = {
        "reactions": [{"product": "CCO", "reactants": ["not-smiles"], "probability": 0.8}],
        "num_steps": 1,
        "step_probability_min": 0.8,
        "step_probability_product": 0.8,
    }
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_retrochimera_strict_rejects_invalid_target_smiles(raw_retrochimera_route) -> None:
    raw_payload = raw_output(raw_retrochimera_route)
    raw_payload["smiles"] = "not-smiles"
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(InvalidSmilesError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, mode="strict")

    assert exc_info.value.code == "chem.invalid_smiles"


@pytest.mark.contract
def test_retrochimera_prune_rejects_invalid_target_smiles(raw_retrochimera_route) -> None:
    raw_payload = raw_output(raw_retrochimera_route)
    raw_payload["smiles"] = "not-smiles"
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_payload)).payload

    with pytest.raises(AdapterLogicError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_retrochimera_strict_rejects_invalid_intermediate_product_smiles() -> None:
    route = {
        "reactions": [
            {"product": "CCO", "reactants": ["C", "not-smiles"], "probability": 0.8},
            {"product": "not-smiles", "reactants": ["C"], "probability": 0.7},
        ],
        "num_steps": 2,
        "step_probability_min": 0.7,
        "step_probability_product": 0.56,
    }
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload

    with pytest.raises(InvalidSmilesError) as exc_info:
        RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"), mode="strict")

    assert exc_info.value.code == "chem.invalid_smiles"


@pytest.mark.contract
def test_retrochimera_prune_skips_invalid_intermediate_product_smiles() -> None:
    route = {
        "reactions": [
            {"product": "CCO", "reactants": ["C", "not-smiles"], "probability": 0.8},
            {"product": "not-smiles", "reactants": ["C"], "probability": 0.7},
        ],
        "num_steps": 2,
        "step_probability_min": 0.7,
        "step_probability_product": 0.56,
    }
    raw_route = next(RetroChimeraAdapter().iter_raw_routes(raw_output(route))).payload

    route = RetroChimeraAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C"]


@pytest.mark.contract
def test_retrochimera_iter_raw_routes_rejects_malformed_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(RetroChimeraAdapter().iter_raw_routes({"result": {}}))
    assert exc_info.value.code == "adapter.schema_invalid"
