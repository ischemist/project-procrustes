import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.models.route import InChIKeyLevel, Molecule, Reaction, Route

KEY_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
KEY_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"
KEY_C = "CCCCCCCCCCCCCC-UHFFFAOYSA-N"
KEY_D = "DDDDDDDDDDDDDD-UHFFFAOYSA-N"
KEY_E = "EEEEEEEEEEEEEE-UHFFFAOYSA-N"
KEY_A_STEREO_1 = "AAAAAAAAAAAAAA-BBBBBBBBSA-N"
KEY_A_STEREO_2 = "AAAAAAAAAAAAAA-CCCCCCCCSA-N"


def molecule(smiles: str, inchikey: str, product_of: Reaction | None = None) -> Molecule:
    return Molecule(smiles=SmilesStr(smiles), inchikey=InChIKeyStr(inchikey), product_of=product_of)


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
    assert root_reaction.id() == "rc:r:/"
    assert intermediate.value.inchikey == KEY_B
    assert child_reaction.product() == intermediate
    assert child_reaction.reactants()[0].id() == "rc:m:/0/0"


@pytest.mark.unit
def test_missing_molecule_or_reaction_lookup_raises_key_error() -> None:
    route = one_step_route([molecule("C", KEY_A)])

    with pytest.raises(ValueError):
        route.molecule_at("rc:r:/")
    with pytest.raises(ValueError):
        route.reaction_at("rc:m:/")
    with pytest.raises(KeyError):
        route.molecule_at("rc:m:/9")
    with pytest.raises(KeyError) as exc_info:
        route.reaction_at("rc:r:/0")
    assert exc_info.value.args == ("rc:r:/0",)


@pytest.mark.unit
def test_lookup_past_leaf_raises_key_error() -> None:
    route = one_step_route([molecule("C", KEY_A)])

    with pytest.raises(KeyError):
        route.molecule_at("rc:m:/0/0")


@pytest.mark.unit
def test_reaction_lookup_past_leaf_raises_requested_reaction_id() -> None:
    route = one_step_route([molecule("C", KEY_A)])

    with pytest.raises(KeyError) as exc_info:
        route.reaction_at("rc:r:/0/0")
    assert exc_info.value.args == ("rc:r:/0/0",)


@pytest.mark.unit
def test_leaf_route_has_target_but_no_root_reaction() -> None:
    route = Route(target=molecule("C", KEY_A))

    assert route.molecule_at("rc:m:/").value.inchikey == KEY_A
    assert route.molecule_at("rc:m:/").produced_by() is None
    assert [leaf.id() for leaf in route.leaves()] == ["rc:m:/"]
    assert route.depth() == 0
    with pytest.raises(KeyError):
        route.reaction_at("rc:r:/")


@pytest.mark.unit
def test_route_leaves_and_depth_use_route_paths() -> None:
    route = two_step_route()

    assert [leaf.id() for leaf in route.leaves()] == ["rc:m:/0/0", "rc:m:/1"]
    assert route.depth() == 2


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
def test_reactant_order_does_not_affect_reaction_signature() -> None:
    route_a = one_step_route([molecule("C", KEY_A), molecule("O", KEY_B)])
    route_b = one_step_route([molecule("O", KEY_B), molecule("C", KEY_A)])

    assert route_a.reaction_at("rc:r:/").signature() == route_b.reaction_at("rc:r:/").signature()


@pytest.mark.unit
def test_duplicate_reactants_do_affect_route_signature() -> None:
    route_a = one_step_route([molecule("C", KEY_A)])
    route_b = one_step_route([molecule("C", KEY_A), molecule("C", KEY_A)])

    assert route_a.signature() != route_b.signature()


@pytest.mark.unit
def test_duplicate_reactants_do_affect_reaction_signature() -> None:
    route_a = one_step_route([molecule("C", KEY_A)])
    route_b = one_step_route([molecule("C", KEY_A), molecule("C", KEY_A)])

    assert route_a.reaction_at("rc:r:/").signature() != route_b.reaction_at("rc:r:/").signature()


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
    assert route.molecule_at("rc:m:/0").value.signature() == molecule("CO", KEY_B).signature()
    assert route.molecule_at("rc:m:/0").subtree_key(depth=0) == ("mol", KEY_B)


@pytest.mark.unit
def test_equal_subtrees_in_different_positions_have_equal_signatures() -> None:
    left = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("C", KEY_A)]))
    right = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("C", KEY_A)]))
    route = one_step_route([left, right])

    assert route.molecule_at("rc:m:/0").id() != route.molecule_at("rc:m:/1").id()
    assert route.molecule_at("rc:m:/0").subtree_signature() == route.molecule_at("rc:m:/1").subtree_signature()


@pytest.mark.unit
def test_route_serialization_does_not_include_derived_node_ids() -> None:
    route = two_step_route()

    payload = route.model_dump_json()
    loaded = Route.model_validate_json(payload)

    assert "rc:m:" not in payload
    assert "rc:r:" not in payload
    assert loaded.signature() == route.signature()


@pytest.mark.unit
def test_route_schema_version_is_literal_v2() -> None:
    route = two_step_route()

    assert route.schema_version == "2"
    with pytest.raises(ValidationError):
        Route.model_validate({"target": route.target.model_dump(mode="json"), "schema_version": "3"})


@pytest.mark.unit
def test_match_level_controls_molecule_identity() -> None:
    route_a = Route(target=molecule("C", KEY_A_STEREO_1))
    route_b = Route(target=molecule("C", KEY_A_STEREO_2))

    assert route_a.signature(InChIKeyLevel.FULL) != route_b.signature(InChIKeyLevel.FULL)
    assert route_a.signature(InChIKeyLevel.NO_STEREO) == route_b.signature(InChIKeyLevel.NO_STEREO)
    assert route_a.signature(InChIKeyLevel.CONNECTIVITY) == route_b.signature(InChIKeyLevel.CONNECTIVITY)


@pytest.mark.unit
def test_negative_depth_is_rejected() -> None:
    route = two_step_route()

    with pytest.raises(ValueError):
        route.key(depth=-1)
    with pytest.raises(ValueError):
        route.signature(depth=-1)
    with pytest.raises(ValueError):
        route.molecule_at("rc:m:/0").subtree_key(depth=-1)
    with pytest.raises(ValueError):
        route.molecule_at("rc:m:/0").subtree_signature(depth=-1)


@pytest.mark.unit
@given(st.permutations([KEY_A, KEY_B, KEY_D]))
def test_generated_reactant_permutations_do_not_change_signature(reactant_keys: tuple[str, ...]) -> None:
    reference = one_step_route([molecule("C", KEY_A), molecule("O", KEY_B), molecule("N", KEY_D)])
    permuted = one_step_route([molecule("C", key) for key in reactant_keys])

    assert permuted.signature() == reference.signature()


@pytest.mark.unit
@given(st.lists(st.sampled_from([KEY_A, KEY_B, KEY_D]), min_size=1, max_size=5))
def test_generated_routes_keep_signature_across_serialization(reactant_keys: list[str]) -> None:
    route = one_step_route([molecule("C", key) for key in reactant_keys])

    loaded = Route.model_validate(route.model_dump(mode="json"))

    assert loaded.signature() == route.signature()


@pytest.mark.unit
@given(st.lists(st.sampled_from([KEY_A, KEY_B, KEY_D]), min_size=1, max_size=5))
def test_generated_duplicate_multiplicity_changes_signature(reactant_keys: list[str]) -> None:
    route = one_step_route([molecule("C", key) for key in reactant_keys])
    with_extra_duplicate = one_step_route([molecule("C", key) for key in [*reactant_keys, reactant_keys[0]]])

    assert with_extra_duplicate.signature() != route.signature()
