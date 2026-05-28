import pytest
from pydantic import ValidationError

from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models.candidates import Candidate, FailureRecord
from retrocast.v2.models.route import Molecule, Reaction, Route


def one_step_route() -> Route:
    leaf = Molecule(smiles=SmilesStr("C"), inchikey=InChIKeyStr("AAAAAAAAAAAAAA-UHFFFAOYSA-N"))
    target = Molecule(
        smiles=SmilesStr("CC"),
        inchikey=InChIKeyStr("BBBBBBBBBBBBBB-UHFFFAOYSA-N"),
        product_of=Reaction(reactants=[leaf]),
    )
    return Route(target=target)


@pytest.mark.unit
def test_candidate_accepts_failure_record() -> None:
    failure = FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="target-1")

    candidate = Candidate(rank=1, failure=failure)

    assert candidate.rank == 1
    assert candidate.failure == failure
    assert candidate.route is None


@pytest.mark.unit
def test_candidate_requires_route_or_failure() -> None:
    with pytest.raises(ValidationError):
        Candidate(rank=1)


@pytest.mark.unit
def test_candidate_rejects_both_route_and_failure() -> None:
    failure = FailureRecord(code=ErrorCode("adapter.schema_invalid"), target_id="target-1")

    with pytest.raises(ValidationError):
        Candidate(rank=1, route=one_step_route(), failure=failure)
