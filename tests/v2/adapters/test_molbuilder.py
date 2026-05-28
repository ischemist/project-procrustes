from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.molbuilder import MolBuilderAdapter
from retrocast.v2.models.task import Target
from tests.v2.adapters.base import (
    AdapterContractCase,
    AdapterContractSuite,
    CastContractCase,
    InvalidSmilesContractCase,
    RawExtractionContractCase,
)

# SECTION: Fixtures


def target_for(smiles: str, target_id: str = "molbuilder-target") -> Target:
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_molbuilder_route() -> dict:
    return {
        "smiles": "CCO",
        "is_purchasable": False,
        "functional_groups": ["alcohol"],
        "best_disconnection": {
            "reaction_name": "Reduction",
            "named_reaction": "NaBH4 Reduction",
            "category": "reduction",
            "score": 0.85,
            "precursors": [{"smiles": "CC=O", "name": "acetaldehyde", "cost_per_kg": 15.0}],
        },
        "children": [{"smiles": "CC=O", "is_purchasable": True, "children": []}, {"smiles": "[H][H]", "children": []}],
    }


@pytest.fixture
def raw_molbuilder_payload(raw_molbuilder_route) -> list[dict]:
    return [raw_molbuilder_route, {"smiles": "CCC", "is_purchasable": True, "children": []}]


@pytest.fixture
def raw_molbuilder_invalid_leaf_route() -> dict:
    return {
        "smiles": "CCO",
        "best_disconnection": {"reaction_name": "Reduction"},
        "children": [{"smiles": "C", "children": []}, {"smiles": "not-smiles", "children": []}],
    }


# SECTION: Shared Contract Suite


class TestMolBuilderAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self,
        raw_molbuilder_payload,
        raw_molbuilder_route,
        raw_molbuilder_invalid_leaf_route,
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=MolBuilderAdapter(),
            extraction=RawExtractionContractCase(
                valid_payload=raw_molbuilder_payload,
                malformed_payload={"smiles": "CCO"},
                source_key="molbuilder-run-1",
                expected_entry_count=2,
                expected_source_keys=["molbuilder-run-1", "molbuilder-run-1"],
                expected_source_order=1,
            ),
            casting=CastContractCase(
                valid_raw_route=raw_molbuilder_route,
                malformed_raw_route={"children": []},
                target=target_for("CCO"),
                mismatched_target=Target(
                    id="molbuilder-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")
                ),
                expected_root_reactant_count=2,
            ),
            invalid_smiles=InvalidSmilesContractCase(
                invalid_leaf_raw_route=raw_molbuilder_invalid_leaf_route,
                expected_pruned_root_reactants=["C"],
            ),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_molbuilder_preserves_route_molecule_and_reaction_annotations(raw_molbuilder_route) -> None:
    route = MolBuilderAdapter().cast(raw_molbuilder_route, target=target_for("CCO"))
    reaction = route.reaction_at("rc:r:/").value

    assert route.annotations["score"] == 0.85
    assert route.target.annotations == {"functional_groups": ["alcohol"]}
    assert reaction.template == "Reduction"
    assert reaction.annotations["reaction_name"] == "Reduction"
    assert reaction.annotations["named_reaction"] == "NaBH4 Reduction"
    assert reaction.annotations["category"] == "reduction"
    assert reaction.annotations["precursors"] == [{"smiles": "CC=O", "name": "acetaldehyde", "cost_per_kg": 15.0}]


@pytest.mark.contract
def test_molbuilder_iter_raw_routes_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(MolBuilderAdapter().iter_raw_routes({"not": "a list"}, source_key="bad"))

    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_molbuilder_rejects_cycles_after_canonicalization() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "OCC", "children": []}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        MolBuilderAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_molbuilder_allows_duplicate_leaf_molecules() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "C", "children": []}, {"smiles": "C", "children": []}]}

    route = MolBuilderAdapter().cast(raw_route, target=target_for("CCO"))

    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_molbuilder_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {"smiles": "CCO", "children": [{"smiles": "not-smiles", "children": []}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        MolBuilderAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"
