import pytest
from pydantic import TypeAdapter, ValidationError

from retrocast.v2.models.route import MoleculeId, ReactionId, RoutePath


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
def test_route_path_navigation_semantics() -> None:
    molecule_path = RoutePath.parse("rc:m:/1/0")
    reaction_path = RoutePath.parse("rc:r:/1/0")

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


@pytest.mark.unit
def test_reaction_and_molecule_id_validators_enforce_kind() -> None:
    assert TypeAdapter(ReactionId).validate_python("rc:r:/0") == "rc:r:/0"
    assert TypeAdapter(MoleculeId).validate_python("rc:m:/0") == "rc:m:/0"

    with pytest.raises(ValidationError):
        TypeAdapter(ReactionId).validate_python("rc:m:/0")
    with pytest.raises(ValidationError):
        TypeAdapter(MoleculeId).validate_python("rc:r:/0")
