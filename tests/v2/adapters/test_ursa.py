from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.ursa import UrsaAdapter
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
    return Target(id="ursa-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


def completion(*reactants: str, product: str = "CCO") -> str:
    reactant_blocks = "".join(f"<reactant><smiles>{reactant}</smiles></reactant>" for reactant in reactants)
    return f"<synthesis_step><product><smiles>{product}</smiles></product>{reactant_blocks}</synthesis_step>"


@pytest.fixture
def raw_ursa_payload() -> list[dict]:
    return [
        {"completion": completion("C", "CC"), "meta": {"product_smiles": "CCO"}},
        {"completion": completion("C"), "meta": {"product_smiles": "CCO"}},
    ]


@pytest.fixture
def ursa_route_payload(raw_ursa_payload):
    return next(UrsaAdapter().iter_raw_routes(raw_ursa_payload, source_key="ursa-run")).payload


@pytest.fixture
def ursa_invalid_leaf_payload():
    return completion("C", "not-smiles")


class TestUrsaAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_ursa_payload, ursa_route_payload, ursa_invalid_leaf_payload
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=UrsaAdapter(),
            extraction=RawExtractionContractCase(
                raw_ursa_payload, [{"completion": 123}], "ursa-run", 2, ["ursa-run", "ursa-run"], 1
            ),
            casting=CastContractCase(
                ursa_route_payload,
                {"not": "completion"},
                target_for("CCO"),
                Target(id="ursa-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(ursa_invalid_leaf_payload, ["C"]),
        )


@pytest.mark.contract
def test_ursa_iter_raw_routes_preserves_row_index_and_target_hint(raw_ursa_payload) -> None:
    entries = list(UrsaAdapter().iter_raw_routes(raw_ursa_payload, source_key="ursa-run"))
    assert [entry.source_row_index for entry in entries] == [1, 2]
    assert [entry.target_hint_smiles for entry in entries] == ["CCO", "CCO"]


@pytest.mark.contract
def test_ursa_iter_raw_routes_allows_missing_target_hint() -> None:
    entries = list(UrsaAdapter().iter_raw_routes([{"completion": completion("C")}]))

    assert entries[0].target_hint_smiles is None


@pytest.mark.contract
def test_ursa_iter_raw_routes_rejects_invalid_meta_product() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(UrsaAdapter().iter_raw_routes([{"completion": completion("C"), "meta": {"product_smiles": "bad"}}]))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_ursa_rejects_missing_target() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        UrsaAdapter().cast(completion("C"))
    assert exc_info.value.code == "adapter.route_transform_failed"


@pytest.mark.contract
def test_ursa_rejects_completion_without_target_step() -> None:
    with pytest.raises(AdapterLogicError) as exc_info:
        UrsaAdapter().cast(completion("C", product="CCC"), target=target_for("CCO"))
    assert exc_info.value.code == "adapter.target_mismatch"


@pytest.mark.contract
def test_ursa_strict_rejects_invalid_product_smiles() -> None:
    with pytest.raises(InvalidSmilesError) as exc_info:
        UrsaAdapter().cast(completion("C", product="not-smiles"), target=target_for("CCO"))
    assert exc_info.value.code == "chem.invalid_smiles"


@pytest.mark.contract
def test_ursa_parses_tokenized_smiles() -> None:
    raw_completion = "<synthesis_step><product><smiles><sm_C><sm_C><sm_O></smiles></product><reactant><smiles><sm_C></smiles></reactant></synthesis_step>"
    route = UrsaAdapter().cast(raw_completion, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C"]


@pytest.mark.contract
def test_ursa_rejects_cycles_after_canonicalization() -> None:
    raw_completion = completion("OCC")
    with pytest.raises(AdapterLogicError) as exc_info:
        UrsaAdapter().cast(raw_completion, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"
