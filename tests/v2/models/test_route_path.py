from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, TypeAdapter, ValidationError

from retrocast.v2.models.route import MoleculeId, ReactionId, RouteNodeKind, RoutePath


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "kind", "indices"),
    [
        ("rc:m:/", "m", ()),
        ("rc:r:/", "r", ()),
        ("rc:m:/1/0", "m", (1, 0)),
        ("rc:r:/1/0", "r", (1, 0)),
    ],
)
def test_route_path_parse_render_round_trip(value: str, kind: str, indices: tuple[int, ...]) -> None:
    path = RoutePath.parse(value)

    assert path.kind == kind
    assert path.indices == indices
    assert path.id() == value
    assert RoutePath.parse(path.id()) == path


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "m:/",
        "rcm/",
        "rc:",
        "rc:x:/",
        "rc:m:",
        "rc:m:0",
        "rc:m:/-1",
        "rc:m:/a",
        "rc:m:/0//1",
        "rc:m:/0/",
    ],
)
def test_route_path_rejects_invalid_grammar(value: str) -> None:
    with pytest.raises(ValueError):
        RoutePath.parse(value)


@pytest.mark.unit
@given(kind=st.sampled_from(["m", "r"]), indices=st.lists(st.integers(min_value=0, max_value=8), max_size=5))
def test_route_path_generated_paths_round_trip(kind: str, indices: list[int]) -> None:
    path = RoutePath(kind="m" if kind == "m" else "r", indices=tuple(indices))

    assert RoutePath.parse(path.id()) == path


@pytest.mark.unit
def test_route_path_direct_construction_validates_invariants() -> None:
    with pytest.raises(ValueError):
        RoutePath(kind=cast(RouteNodeKind, "x"))
    with pytest.raises(ValueError):
        RoutePath(kind="m", indices=(-1,))


@pytest.mark.unit
def test_route_path_navigation_semantics() -> None:
    molecule_path = RoutePath.parse("rc:m:/1/0")
    reaction_path = RoutePath.parse("rc:r:/1/0")

    assert molecule_path.depth() == 2
    assert RoutePath.target().id() == "rc:m:/"
    assert RoutePath.root_reaction().id() == "rc:r:/"
    assert molecule_path.produced_by().id() == "rc:r:/1/0"
    assert reaction_path.product().id() == "rc:m:/1/0"
    assert reaction_path.reactant(2).id() == "rc:m:/1/0/2"


@pytest.mark.unit
def test_route_path_rejects_wrong_navigation_kind() -> None:
    with pytest.raises(ValueError):
        RoutePath.root_reaction().produced_by()
    with pytest.raises(ValueError):
        RoutePath.target().product()
    with pytest.raises(ValueError):
        RoutePath.target().reactant(0)
    with pytest.raises(ValueError):
        RoutePath.root_reaction().reactant(-1)


@pytest.mark.unit
def test_reaction_and_molecule_id_validators_enforce_kind() -> None:
    assert TypeAdapter(ReactionId).validate_python("rc:r:/0") == "rc:r:/0"
    assert TypeAdapter(MoleculeId).validate_python("rc:m:/0") == "rc:m:/0"

    with pytest.raises(ValidationError):
        TypeAdapter(ReactionId).validate_python("rc:m:/0")
    with pytest.raises(ValidationError):
        TypeAdapter(MoleculeId).validate_python("rc:r:/0")


@pytest.mark.unit
def test_reaction_and_molecule_id_validators_work_inside_models() -> None:
    class NodeIds(BaseModel):
        reaction_id: ReactionId
        molecule_id: MoleculeId

    ids = NodeIds(reaction_id="rc:r:/1", molecule_id="rc:m:/1")

    assert ids.reaction_id == "rc:r:/1"
    assert ids.molecule_id == "rc:m:/1"

    with pytest.raises(ValidationError):
        NodeIds(reaction_id="rc:m:/1", molecule_id="rc:m:/1")
