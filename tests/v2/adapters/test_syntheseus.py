from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.syntheseus import SyntheseusAdapter
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
    return Target(id="syntheseus-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_syntheseus_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"template": "tmpl", "mapped_reaction_smiles": "C.C>>CCO"},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True},
                    {"type": "mol", "smiles": "CC", "in_stock": True},
                ],
            }
        ],
    }


@pytest.fixture
def raw_syntheseus_payload(raw_syntheseus_route) -> list[dict]:
    return [raw_syntheseus_route, {"type": "mol", "smiles": "CCC", "in_stock": True}]


@pytest.fixture
def raw_syntheseus_invalid_leaf_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [{"type": "mol", "smiles": "C"}, {"type": "mol", "smiles": "not-smiles"}],
            }
        ],
    }


# SECTION: Shared Contract Suite


class TestSyntheseusAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_syntheseus_payload, raw_syntheseus_route, raw_syntheseus_invalid_leaf_route
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=SyntheseusAdapter(),
            extraction=RawExtractionContractCase(
                raw_syntheseus_payload, {"type": "mol"}, "syntheseus-run", 2, ["syntheseus-run", "syntheseus-run"], 1
            ),
            casting=CastContractCase(
                raw_syntheseus_route,
                {"type": "mol"},
                target_for("CCO"),
                Target(id="syntheseus-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(raw_syntheseus_invalid_leaf_route, ["C"]),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_syntheseus_preserves_reaction_metadata(raw_syntheseus_route) -> None:
    reaction = SyntheseusAdapter().cast(raw_syntheseus_route, target=target_for("CCO")).reaction_at("rc:r:/").value
    assert reaction.template == "tmpl"
    assert reaction.mapped_reaction_smiles == "C.C>>CCO"
    assert reaction.annotations["template"] == "tmpl"


@pytest.mark.contract
def test_syntheseus_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(SyntheseusAdapter().iter_raw_routes({"not": "a list"}))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_syntheseus_rejects_cycles_after_canonicalization() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [{"type": "reaction", "smiles": "CCO", "children": [{"type": "mol", "smiles": "OCC"}]}],
    }
    with pytest.raises(AdapterLogicError) as exc_info:
        SyntheseusAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_syntheseus_allows_duplicate_leaf_molecules() -> None:
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
    route = SyntheseusAdapter().cast(raw_route, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_syntheseus_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [{"type": "mol", "smiles": "not-smiles"}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        SyntheseusAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"
