from __future__ import annotations

import pytest

from retrocast.adapters.aizynth import AiZynthFinderAdapter
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
    canon_smiles = canonicalize_smiles(smiles)
    return Target(id="aizynth-target", smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))


@pytest.fixture
def raw_aizynth_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "scores": {"state score": 0.75},
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"template": "tmpl-1", "mapped_reaction_smiles": "CC=O.[H][H]>>CCO"},
                "children": [
                    {"type": "mol", "smiles": "CC=O", "in_stock": True},
                    {"type": "mol", "smiles": "[H][H]", "in_stock": True},
                ],
            }
        ],
    }


@pytest.fixture
def raw_aizynth_payload(raw_aizynth_route) -> list[dict]:
    return [raw_aizynth_route, {"type": "mol", "smiles": "CCC", "in_stock": True}]


@pytest.fixture
def raw_aizynth_invalid_leaf_route() -> dict:
    return {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True},
                    {"type": "mol", "smiles": "not-smiles", "in_stock": True},
                ],
            }
        ],
    }


# SECTION: Shared Contract Suite


class TestAiZynthAdapterContract(AdapterContractSuite):
    @pytest.fixture
    def adapter_contract_case(
        self, raw_aizynth_payload, raw_aizynth_route, raw_aizynth_invalid_leaf_route
    ) -> AdapterContractCase:
        return AdapterContractCase(
            adapter=AiZynthFinderAdapter(),
            extraction=RawExtractionContractCase(
                raw_aizynth_payload, {"type": "mol"}, "aizynth-run", 2, ["aizynth-run", "aizynth-run"], 1
            ),
            casting=CastContractCase(
                raw_aizynth_route,
                {"type": "mol"},
                target_for("CCO"),
                Target(id="aizynth-target", smiles=SmilesStr("CCC"), inchikey=get_inchi_key("CCC")),
                2,
            ),
            invalid_smiles=InvalidSmilesContractCase(raw_aizynth_invalid_leaf_route, ["C"]),
        )


# SECTION: Contract Tests


@pytest.mark.contract
def test_aizynth_preserves_scores_and_reaction_metadata(raw_aizynth_route) -> None:
    route = AiZynthFinderAdapter().cast(raw_aizynth_route, target=target_for("CCO"))
    reaction = route.reaction_at("rc:r:/").value

    assert route.annotations == {"scores": {"state score": 0.75}, "state_score": 0.75}
    assert reaction.template == "tmpl-1"
    assert reaction.mapped_reaction_smiles == "CC=O.[H][H]>>CCO"
    assert reaction.annotations["template"] == "tmpl-1"


@pytest.mark.contract
def test_aizynth_rejects_non_list_payload() -> None:
    with pytest.raises(AdapterSchemaError) as exc_info:
        list(AiZynthFinderAdapter().iter_raw_routes({"not": "a list"}))
    assert exc_info.value.code == "adapter.schema_invalid"


@pytest.mark.contract
def test_aizynth_rejects_cycles_after_canonicalization() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [{"type": "reaction", "smiles": "CCO", "children": [{"type": "mol", "smiles": "OCC"}]}],
    }
    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))
    assert exc_info.value.code == "adapter.cycle_detected"


@pytest.mark.contract
def test_aizynth_allows_duplicate_leaf_molecules() -> None:
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
    route = AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["C", "C"]


@pytest.mark.contract
def test_aizynth_prune_rejects_route_when_all_reactants_are_invalid() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [{"type": "mol", "smiles": "not-smiles", "in_stock": True}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"), mode="prune")

    assert exc_info.value.code == "adapter.target_pruned"


@pytest.mark.contract
def test_aizynth_rejects_empty_reaction_in_strict_mode() -> None:
    raw_route = {"type": "mol", "smiles": "CCO", "children": [{"type": "reaction", "smiles": "CCO"}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.reaction_empty"


@pytest.mark.contract
def test_aizynth_rejects_multiple_child_reactions() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {"type": "reaction", "smiles": "CCO", "children": [{"type": "mol", "smiles": "C"}]},
            {"type": "reaction", "smiles": "CCO", "children": [{"type": "mol", "smiles": "CC"}]},
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.route_not_tree"


@pytest.mark.contract
def test_aizynth_rejects_molecule_child_under_molecule() -> None:
    raw_route = {"type": "mol", "smiles": "CCO", "children": [{"type": "mol", "smiles": "C"}]}

    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.node_type_invalid"


@pytest.mark.contract
def test_aizynth_rejects_reaction_child_under_reaction() -> None:
    raw_route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "children": [{"type": "reaction", "smiles": "C>>CCO"}],
            }
        ],
    }

    with pytest.raises(AdapterLogicError) as exc_info:
        AiZynthFinderAdapter().cast(raw_route, target=target_for("CCO"))

    assert exc_info.value.code == "adapter.node_type_invalid"
