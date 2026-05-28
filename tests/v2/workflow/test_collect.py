from __future__ import annotations

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models.candidates import Candidate, FailureRecord
from retrocast.v2.models.route import Molecule, Route
from retrocast.v2.models.task import Target, Task
from retrocast.v2.workflow.collect import collect_candidates, collect_routes


def target(smiles: str, target_id: str) -> Target:
    canonical = canonicalize_smiles(smiles)
    return Target(id=target_id, smiles=SmilesStr(canonical), inchikey=get_inchi_key(canonical))


def route(smiles: str) -> Route:
    canonical = canonicalize_smiles(smiles)
    return Route(target=Molecule(smiles=SmilesStr(canonical), inchikey=InChIKeyStr(get_inchi_key(canonical))))


def task() -> Task:
    first = target("CCO", "ethanol")
    second = target("CC(=O)O", "acetic-acid")
    return Task(name="two-targets", targets={first.id: first, second.id: second})


def test_collect_candidates_places_successful_candidate_by_route_target() -> None:
    candidate = Candidate(rank=1, route=route("CCO"))

    collected = collect_candidates([candidate], task())

    assert collected["ethanol"] == [candidate]
    assert collected["acetic-acid"] == []


def test_collect_candidates_places_failed_candidate_by_target_id() -> None:
    candidate = Candidate(rank=1, failure=FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="ethanol"))

    collected = collect_candidates([candidate], task())

    assert collected["ethanol"] == [candidate]
    assert collected["acetic-acid"] == []


def test_collect_candidates_places_failed_candidate_by_target_inchikey() -> None:
    candidate = Candidate(
        rank=1,
        failure=FailureRecord(
            code=ErrorCode("adapter.schema_invalid"),
            target_inchikey=InChIKeyStr(get_inchi_key("CC(=O)O")),
        ),
    )

    collected = collect_candidates([candidate], task())
    assert collected["ethanol"] == []
    assert collected["acetic-acid"] == [candidate]


def test_collect_routes_places_routes_by_target() -> None:
    ethanol_route = route("CCO")
    unmatched_route = route("CCC")

    collected = collect_routes([ethanol_route, unmatched_route], task())

    assert collected["ethanol"] == [ethanol_route]
    assert collected["acetic-acid"] == []
