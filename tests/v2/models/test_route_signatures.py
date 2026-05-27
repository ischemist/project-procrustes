import pytest

from retrocast.typing import InchiKeyStr, SmilesStr
from retrocast.v2.models.route import Molecule, Reaction, Route

KEY_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
KEY_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"
KEY_C = "CCCCCCCCCCCCCC-UHFFFAOYSA-N"
KEY_D = "DDDDDDDDDDDDDD-UHFFFAOYSA-N"
KEY_E = "EEEEEEEEEEEEEE-UHFFFAOYSA-N"


def molecule(smiles: str, inchikey: str, product_of: Reaction | None = None) -> Molecule:
    return Molecule(smiles=SmilesStr(smiles), inchikey=InchiKeyStr(inchikey), product_of=product_of)


def one_step_route(reactants: list[Molecule], *, target_key: str = KEY_C) -> Route:
    target = molecule("CC", target_key, product_of=Reaction(reactants=reactants))
    return Route(target=target)


def two_step_route() -> Route:
    leaf = molecule("C", KEY_A)
    intermediate = molecule("CO", KEY_B, product_of=Reaction(reactants=[leaf]))
    target = molecule("CCO", KEY_C, product_of=Reaction(reactants=[intermediate, molecule("N", KEY_D)]))
    return Route(target=target)


@pytest.mark.unit
def test_path_navigation_and_lookup_return_contextual_views() -> None:
    route = two_step_route()

    target = route.molecule_at("rc:m:/")
    root_reaction = route.reaction_at("rc:r:/")
    intermediate = route.molecule_at("rc:m:/0")
    child_reaction = route.reaction_at("rc:r:/0")

    assert target.id() == "rc:m:/"
    assert target.produced_by() == root_reaction
    assert [reactant.id() for reactant in root_reaction.reactants()] == ["rc:m:/0", "rc:m:/1"]
    assert intermediate.value.inchikey == KEY_B
    assert child_reaction.product() == intermediate
    assert child_reaction.reactants()[0].id() == "rc:m:/0/0"


@pytest.mark.unit
def test_missing_molecule_or_reaction_lookup_raises_key_error() -> None:
    route = one_step_route([molecule("C", KEY_A)])

    with pytest.raises(KeyError):
        route.molecule_at("rc:m:/9")
    with pytest.raises(KeyError):
        route.reaction_at("rc:r:/0")


@pytest.mark.unit
def test_reaction_identity_includes_product_and_reactants() -> None:
    route_a = one_step_route([molecule("C", KEY_A), molecule("O", KEY_B)], target_key=KEY_C)
    route_b = one_step_route([molecule("C", KEY_A), molecule("O", KEY_B)], target_key=KEY_D)
    route_c = one_step_route([molecule("C", KEY_A), molecule("N", KEY_E)], target_key=KEY_C)

    assert route_a.reaction_at("rc:r:/").signature() != route_b.reaction_at("rc:r:/").signature()
    assert route_a.reaction_at("rc:r:/").signature() != route_c.reaction_at("rc:r:/").signature()


@pytest.mark.unit
def test_reactant_order_does_not_affect_route_signature() -> None:
    route_a = one_step_route([molecule("C", KEY_A), molecule("O", KEY_B)])
    route_b = one_step_route([molecule("O", KEY_B), molecule("C", KEY_A)])

    assert route_a.signature() == route_b.signature()


@pytest.mark.unit
def test_duplicate_reactants_do_affect_route_signature() -> None:
    route_a = one_step_route([molecule("C", KEY_A)])
    route_b = one_step_route([molecule("C", KEY_A), molecule("C", KEY_A)])

    assert route_a.signature() != route_b.signature()


@pytest.mark.unit
def test_prefix_depth_changes_signatures_as_expected() -> None:
    shallow_intermediate = molecule("CO", KEY_B)
    deep_intermediate = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("C", KEY_A)]))
    shallow_route = one_step_route([shallow_intermediate, molecule("N", KEY_D)])
    deep_route = one_step_route([deep_intermediate, molecule("N", KEY_D)])

    assert shallow_route.signature(depth=0) == deep_route.signature(depth=0)
    assert shallow_route.signature(depth=1) == deep_route.signature(depth=1)
    assert shallow_route.signature() != deep_route.signature()


@pytest.mark.unit
def test_subtree_signature_addresses_non_root_molecules() -> None:
    route = two_step_route()

    assert route.molecule_at("rc:m:/0").subtree_signature() != route.molecule_at("rc:m:/1").subtree_signature()
    assert route.molecule_at("rc:m:/0").subtree_key(depth=0) == ("mol", KEY_B)
