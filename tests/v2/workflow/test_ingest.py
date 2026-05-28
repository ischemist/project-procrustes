from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.models.route import Molecule, Route
from retrocast.v2.models.task import Target, Task
from retrocast.v2.workflow.adapt import adapt_candidates, adapt_routes
from retrocast.v2.workflow.collect import collect_candidates, collect_routes
from retrocast.v2.workflow.ingest import ingest_candidates, ingest_routes


class SmilesAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        for index, raw_route in enumerate(raw_payload, start=1):
            yield RawRouteEntry(payload=raw_route, source_key=source_key, source_order=index)

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        smiles = canonicalize_smiles(raw_route["smiles"])
        return Route(target=Molecule(smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles))))


def target(smiles: str, target_id: str) -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=SmilesStr(canonical), inchikey=get_inchi_key(canonical))


def task() -> Task:
    ethanol = target("CCO", "ethanol")
    return Task(name="one-target", targets={ethanol.id: ethanol})


def test_ingest_routes_is_adapt_plus_collect() -> None:
    raw_payload = [{"smiles": "CCO"}, {"smiles": "not-a-smiles"}]
    adapter = SmilesAdapter()
    expected = collect_routes(adapt_routes(raw_payload, adapter, target=target("CCO", "ethanol")), task())

    assert ingest_routes(raw_payload, adapter, task()) == expected


def test_ingest_candidates_is_adapt_plus_collect() -> None:
    raw_payload = [{"smiles": "CCO"}, {"smiles": "not-a-smiles"}]
    adapter = SmilesAdapter()
    expected = collect_candidates(adapt_candidates(raw_payload, adapter, target=target("CCO", "ethanol")), task())

    collected = ingest_candidates(raw_payload, adapter, task())

    assert collected == expected
    assert [candidate.rank for candidate in collected["ethanol"]] == [1, 2]
    assert collected["ethanol"][1].failure is not None
    assert collected["ethanol"][1].failure.code == ErrorCode("chem.invalid_smiles")


def test_ingest_candidates_handles_target_keyed_payloads() -> None:
    raw_payload = {"ethanol": [{"smiles": "CCO"}, {"smiles": "not-a-smiles"}]}

    collected = ingest_candidates(raw_payload, SmilesAdapter(), task())

    assert [candidate.rank for candidate in collected["ethanol"]] == [1, 2]
    assert collected["ethanol"][0].route is not None
    assert collected["ethanol"][1].failure is not None


def test_ingest_candidates_handles_target_smiles_keyed_payloads() -> None:
    raw_payload = {"CCO": [{"smiles": "CCO"}]}

    collected = ingest_candidates(raw_payload, SmilesAdapter(), task())

    assert len(collected["ethanol"]) == 1
    assert collected["ethanol"][0].route is not None
