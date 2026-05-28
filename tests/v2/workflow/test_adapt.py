from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from retrocast.adapters.errors import adapter_target_mismatch
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, AdapterSchemaError
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.adapters.common import build_plain_tree_molecule
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target
from retrocast.v2.workflow.adapt import adapt_candidates, adapt_route, adapt_routes


class TreeAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        if not isinstance(raw_payload, list):
            raise AdapterSchemaError("expected list", code="adapter.schema_invalid")
        for index, raw_route in enumerate(raw_payload, start=1):
            if not isinstance(raw_route, dict):
                raise AdapterSchemaError("expected route dict", code="adapter.schema_invalid")
            yield RawRouteEntry(
                payload=raw_route,
                source_key=source_key,
                source_row_index=index,
                source_order=raw_route.get("rank"),
                target_hint_id=raw_route.get("target_id"),
                target_hint_smiles=raw_route.get("target_smiles"),
            )

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        if not isinstance(raw_route, dict):
            raise AdapterSchemaError("expected route dict", code="adapter.schema_invalid")
        molecule = build_plain_tree_molecule(
            raw_route,
            adapter="tree",
            mode=mode,
            get_smiles=lambda node: node["smiles"],
            get_children=lambda node: node.get("children", []),
        )
        if molecule is None:
            raise AdapterLogicError("target was pruned", code="adapter.target_pruned")
        if target is not None and molecule.smiles != canonicalize_smiles(target.smiles):
            raise adapter_target_mismatch(
                "tree",
                target.id,
                expected_smiles=target.smiles,
                actual_smiles=molecule.smiles,
            )
        return Route(target=molecule)


class FailingExtractionAdapter(TreeAdapter):
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        yield RawRouteEntry(payload=valid_route(), source_key=source_key)
        raise AdapterSchemaError("bad raw record", code="adapter.schema_invalid")


def make_target(smiles: str = "CC(=O)O", *, target_id: str = "target-1") -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=canonical, inchikey=get_inchi_key(canonical))


def valid_route(rank: int = 1) -> dict[str, Any]:
    return {
        "rank": rank,
        "smiles": "CC(=O)O",
        "children": [{"smiles": "CCO"}, {"smiles": "O"}],
    }


def test_adapt_route_returns_route_for_valid_raw_route() -> None:
    route = adapt_route(valid_route(), TreeAdapter(), target=make_target())

    assert isinstance(route, Route)
    assert route.target.smiles == "CC(=O)O"


def test_adapt_routes_filters_failed_raw_routes() -> None:
    routes = adapt_routes([valid_route(), {"rank": 2, "smiles": "not-a-smiles"}], TreeAdapter(), target=make_target())

    assert len(routes) == 1
    assert routes[0].target.smiles == "CC(=O)O"


def test_adapt_candidates_preserves_rank_and_failed_candidate() -> None:
    candidates = adapt_candidates(
        [valid_route(rank=7), {"rank": 8, "smiles": "not-a-smiles"}],
        TreeAdapter(),
        target=make_target(),
    )

    assert [candidate.rank for candidate in candidates] == [7, 8]
    assert candidates[0].route is not None
    assert candidates[0].failure is None
    assert candidates[1].route is None
    assert candidates[1].failure is not None
    assert candidates[1].failure.code == "chem.invalid_smiles"


def test_adapt_candidates_records_failed_target_identity_from_entry_hint() -> None:
    candidates = adapt_candidates(
        [{"rank": 3, "target_id": "target-1", "target_smiles": "CC(=O)O", "smiles": "not-a-smiles"}],
        TreeAdapter(),
    )

    failure = candidates[0].failure
    assert failure is not None
    assert failure.target_id == "target-1"
    assert failure.target_smiles == "CC(=O)O"
    assert failure.target_inchikey == get_inchi_key("CC(=O)O")


def test_prune_mode_drops_invalid_branch() -> None:
    candidates = adapt_candidates(
        [
            {
                "rank": 1,
                "smiles": "CC(=O)O",
                "children": [{"smiles": "CCO"}, {"smiles": "not-a-smiles"}],
            }
        ],
        TreeAdapter(),
        mode="prune",
        target=make_target(),
    )

    assert len(candidates) == 1
    route = candidates[0].route
    assert route is not None
    assert not hasattr(route, "rank")
    assert [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()] == ["CCO"]


def test_raw_payload_schema_failure_propagates() -> None:
    with pytest.raises(AdapterSchemaError):
        adapt_candidates({"not": "a-list"}, TreeAdapter(), target=make_target())


def test_extraction_failure_after_yield_propagates() -> None:
    with pytest.raises(AdapterSchemaError):
        adapt_candidates([], FailingExtractionAdapter(), target=make_target())
