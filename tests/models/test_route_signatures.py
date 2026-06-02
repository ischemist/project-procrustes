from itertools import permutations
from typing import TypeAlias

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from retrocast.models.route import InChIKeyLevel, Molecule, Reaction, Route
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr

KEY_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
KEY_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"
KEY_C = "CCCCCCCCCCCCCC-UHFFFAOYSA-N"
KEY_D = "DDDDDDDDDDDDDD-UHFFFAOYSA-N"
KEY_E = "EEEEEEEEEEEEEE-UHFFFAOYSA-N"
KEY_A_STEREO_1 = "AAAAAAAAAAAAAA-BBBBBBBBSA-N"
KEY_A_STEREO_2 = "AAAAAAAAAAAAAA-CCCCCCCCSA-N"
KEY_TO_SMILES = {
    KEY_A: "C",
    KEY_B: "CO",
    KEY_C: "CC",
    KEY_D: "N",
    KEY_E: "O",
}
TREE_KEYS = (KEY_A, KEY_B, KEY_D, KEY_E)
TreeShape: TypeAlias = tuple[str, tuple["TreeShape", ...]]


def molecule(smiles: str, inchikey: str, product_of: Reaction | None = None) -> Molecule:
    return Molecule(smiles=SmilesStr(smiles), inchikey=InChIKeyStr(inchikey), product_of=product_of)


def one_step_route(reactants: list[Molecule], *, target_key: str = KEY_C) -> Route:
    target = molecule("CC", target_key, product_of=Reaction(reactants=reactants))
    return Route(target=target)


def content_route(
    *,
    mapped_reaction_smiles: str | None = None,
    template: str | None = None,
    reagents: list[str] | None = None,
    solvents: list[str] | None = None,
) -> Route:
    reaction = Reaction(
        reactants=[molecule("C", KEY_A), molecule("O", KEY_B)],
        mapped_reaction_smiles=ReactionSmilesStr(mapped_reaction_smiles)
        if mapped_reaction_smiles is not None
        else None,
        template=template,
        reagents=[SmilesStr(reagent) for reagent in reagents] if reagents is not None else None,
        solvents=[SmilesStr(solvent) for solvent in solvents] if solvents is not None else None,
    )
    return Route(target=molecule("CC", KEY_C, product_of=reaction))


def two_step_route() -> Route:
    leaf = molecule("C", KEY_A)
    intermediate = molecule("CO", KEY_B, product_of=Reaction(reactants=[leaf]))
    target = molecule("CCO", KEY_C, product_of=Reaction(reactants=[intermediate, molecule("N", KEY_D)]))
    return Route(target=target)


def route_from_shape(shape: TreeShape) -> Route:
    return Route(target=molecule_from_shape((KEY_C, (shape,))))


def molecule_from_shape(shape: TreeShape) -> Molecule:
    key, children = shape
    reaction = None if not children else Reaction(reactants=[molecule_from_shape(child) for child in children])
    return molecule(KEY_TO_SMILES[key], key, product_of=reaction)


def reverse_child_order(shape: TreeShape) -> TreeShape:
    key, children = shape
    return (key, tuple(reverse_child_order(child) for child in reversed(children)))


def route_path_keys(route: Route) -> dict[str, str]:
    return {molecule.id(): molecule.subtree_signature() for molecule in route.iter_molecules()}


tree_shapes = st.recursive(
    st.sampled_from(TREE_KEYS).map(lambda key: (key, ())),
    lambda children: st.tuples(st.sampled_from(TREE_KEYS), st.lists(children, min_size=1, max_size=3).map(tuple)),
    max_leaves=8,
)


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
    assert [leaf.id() for leaf in route.iter_leaves()] == ["rc:m:/"]
    assert route.depth() == 0
    with pytest.raises(KeyError):
        route.reaction_at("rc:r:/")


@pytest.mark.unit
def test_route_leaves_and_depth_use_route_paths() -> None:
    route = two_step_route()

    assert [leaf.id() for leaf in route.leaves()] == ["rc:m:/0/0", "rc:m:/1"]
    assert [leaf.id() for leaf in route.iter_leaves()] == ["rc:m:/0/0", "rc:m:/1"]
    assert route.depth() == 2


@pytest.mark.unit
def test_find_molecules_returns_all_matching_route_nodes() -> None:
    route = one_step_route([molecule("C", KEY_A), molecule("C", KEY_A), molecule("O", KEY_B)])

    matches = route.find_molecules(molecule("C", KEY_A))

    assert [match.id() for match in matches] == ["rc:m:/0", "rc:m:/1"]
    assert route.contains_molecule(molecule("O", KEY_B)) is True
    assert route.contains_molecule(molecule("N", KEY_E)) is False


@pytest.mark.unit
def test_contains_molecule_uses_requested_match_level() -> None:
    route = Route(target=molecule("C", KEY_A_STEREO_1))

    assert route.contains_molecule(molecule("C", KEY_A_STEREO_2), InChIKeyLevel.FULL) is False
    assert route.contains_molecule(molecule("C", KEY_A_STEREO_2), InChIKeyLevel.NO_STEREO) is True


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
def test_reactant_order_does_not_affect_route_paths() -> None:
    intermediate = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("N", KEY_D), molecule("C", KEY_A)]))
    route_a = one_step_route([molecule("O", KEY_E), intermediate])
    route_b = one_step_route([intermediate, molecule("O", KEY_E)])

    assert route_a.model_dump(mode="json") == route_b.model_dump(mode="json")
    assert route_a.molecule_at("rc:m:/0").value.inchikey == KEY_B
    assert route_b.molecule_at("rc:m:/0").value.inchikey == KEY_B
    assert route_a.molecule_at("rc:m:/0/0").value.inchikey == KEY_A
    assert route_b.molecule_at("rc:m:/0/0").value.inchikey == KEY_A


@pytest.mark.unit
def test_nested_reactant_permutations_normalize_to_same_route_object() -> None:
    def build_route(root_order: tuple[str, ...], child_order: tuple[str, ...]) -> Route:
        leaves = {
            KEY_A: molecule("C", KEY_A),
            KEY_D: molecule("N", KEY_D),
        }
        intermediate = molecule(
            "CO",
            KEY_B,
            product_of=Reaction(reactants=[leaves[key] for key in child_order]),
        )
        root_reactants = {
            KEY_B: intermediate,
            KEY_E: molecule("O", KEY_E),
        }
        return one_step_route([root_reactants[key] for key in root_order])

    routes = [
        build_route(root_order, child_order)
        for root_order in permutations((KEY_B, KEY_E))
        for child_order in permutations((KEY_A, KEY_D))
    ]

    reference = routes[0]
    assert {route.signature() for route in routes} == {reference.signature()}
    assert all(route == reference for route in routes)
    assert all(route.model_dump(mode="json") == reference.model_dump(mode="json") for route in routes)


@pytest.mark.unit
def test_duplicate_identity_reactants_use_stable_tiebreaker_for_paths() -> None:
    left = molecule("C", KEY_A, product_of=Reaction(reactants=[molecule("N", KEY_D)]))
    right = molecule("C", KEY_A, product_of=Reaction(reactants=[molecule("O", KEY_E)]))

    route_a = one_step_route([left, right])
    route_b = one_step_route([right, left])

    assert route_a.signature() == route_b.signature()
    assert route_a == route_b
    assert route_a.molecule_at("rc:m:/0/0").value.inchikey == KEY_D
    assert route_b.molecule_at("rc:m:/0/0").value.inchikey == KEY_D


@pytest.mark.unit
def test_reactant_order_tiebreaker_runs_only_for_structural_collisions() -> None:
    route = one_step_route(
        [
            molecule("C", KEY_A).model_copy(update={"annotations": {"opaque": object()}}),
            molecule("O", KEY_B),
        ]
    )

    assert [reactant.value.inchikey for reactant in route.reaction_at("rc:r:/").reactants()] == [KEY_A, KEY_B]


@pytest.mark.unit
def test_reactant_order_tiebreaker_ignores_annotations() -> None:
    first = molecule(
        "C",
        KEY_A,
        product_of=Reaction(
            reactants=[molecule("N", KEY_D)],
            template="template-a",
            annotations={"opaque": object()},
        ),
    ).model_copy(update={"annotations": {"opaque": object()}})
    second = molecule(
        "C",
        KEY_A,
        product_of=Reaction(
            reactants=[molecule("N", KEY_D)],
            template="template-b",
            annotations={"opaque": object()},
        ),
    ).model_copy(update={"annotations": {"opaque": object()}})

    route = one_step_route([second, first])
    first_reactant_reaction = route.molecule_at("rc:m:/0").value.product_of

    assert first_reactant_reaction is not None
    assert first_reactant_reaction.template == "template-a"


@pytest.mark.unit
@given(tree_shapes)
def test_generated_recursive_reactant_permutations_normalize_to_same_route(shape: TreeShape) -> None:
    route = route_from_shape(shape)
    permuted = route_from_shape(reverse_child_order(shape))

    assert permuted.signature() == route.signature()
    assert permuted.model_dump(mode="json") == route.model_dump(mode="json")
    assert route_path_keys(permuted) == route_path_keys(route)


@pytest.mark.unit
@given(st.lists(tree_shapes, min_size=1, max_size=5))
def test_generated_sibling_permutations_keep_route_paths_stable(children: list[TreeShape]) -> None:
    route = route_from_shape((KEY_C, tuple(children)))
    permuted = route_from_shape((KEY_C, tuple(reversed(children))))

    assert permuted.signature() == route.signature()
    assert route_path_keys(permuted) == route_path_keys(route)


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
def test_structural_signature_ignores_reaction_content() -> None:
    route_a = content_route(mapped_reaction_smiles="C.O>>CC", template="template-a", reagents=["N"], solvents=["O"])
    route_b = content_route(mapped_reaction_smiles="C.N>>CC", template="template-b", reagents=["C"], solvents=["CO"])

    assert route_a.signature() == route_b.signature()
    assert route_a.reaction_at("rc:r:/").signature() == route_b.reaction_at("rc:r:/").signature()


@pytest.mark.unit
def test_content_signature_uses_only_selected_reaction_fields() -> None:
    route_a = content_route(mapped_reaction_smiles="C.O>>CC", template="same")
    route_b = content_route(mapped_reaction_smiles="C.N>>CC", template="same")

    assert route_a.content_signature(fields=("mapped_reaction_smiles",)) != route_b.content_signature(
        fields=("mapped_reaction_smiles",)
    )
    assert route_a.content_signature(fields=("template",)) == route_b.content_signature(fields=("template",))


@pytest.mark.unit
def test_content_signature_distinguishes_absent_selected_fields() -> None:
    route_a = content_route(template="template-a")
    route_b = content_route(template=None)

    assert route_a.content_signature(fields=("template",)) != route_b.content_signature(fields=("template",))


@pytest.mark.unit
def test_content_signature_treats_reagents_and_solvents_as_unordered() -> None:
    route_a = content_route(reagents=["N", "O"], solvents=["C", "CO"])
    route_b = content_route(reagents=["O", "N"], solvents=["CO", "C"])

    assert route_a.content_signature(fields=("reagents", "solvents")) == route_b.content_signature(
        fields=("solvents", "reagents")
    )


@pytest.mark.unit
def test_content_signature_treats_empty_reagents_and_solvents_as_absent() -> None:
    route_a = content_route(reagents=[], solvents=[])
    route_b = content_route(reagents=None, solvents=None)

    assert route_a.content_signature(fields=("reagents", "solvents")) == route_b.content_signature(
        fields=("reagents", "solvents")
    )


@pytest.mark.unit
def test_empty_content_fields_use_structural_identity() -> None:
    route = content_route(mapped_reaction_smiles="C.O>>CC", template="template-a")
    target = route.molecule_at("rc:m:/")
    reaction = route.reaction_at("rc:r:/")

    assert route.content_signature(fields=()) == route.signature()
    assert route.content_signature(fields=(), depth=1) == route.signature(depth=1)
    assert target.content_subtree_signature(fields=()) == target.subtree_signature()
    assert reaction.content_signature(fields=()) == reaction.signature()
    assert route.reaction_content_signatures(fields=()) == route.reaction_signatures()


@pytest.mark.unit
def test_content_signature_is_reactant_order_invariant() -> None:
    reaction_a = Reaction(reactants=[molecule("C", KEY_A), molecule("O", KEY_B)], template="same")
    reaction_b = Reaction(reactants=[molecule("O", KEY_B), molecule("C", KEY_A)], template="same")
    route_a = Route(target=molecule("CC", KEY_C, product_of=reaction_a))
    route_b = Route(target=molecule("CC", KEY_C, product_of=reaction_b))

    assert route_a.content_signature(fields=("template",)) == route_b.content_signature(fields=("template",))


@pytest.mark.unit
def test_content_signature_uses_requested_match_level() -> None:
    route_a = Route(
        target=molecule(
            "CC",
            KEY_C,
            product_of=Reaction(reactants=[molecule("C", KEY_A_STEREO_1)], template="same"),
        )
    )
    route_b = Route(
        target=molecule(
            "CC",
            KEY_C,
            product_of=Reaction(reactants=[molecule("C", KEY_A_STEREO_2)], template="same"),
        )
    )

    assert route_a.content_signature(InChIKeyLevel.FULL, fields=("template",)) != route_b.content_signature(
        InChIKeyLevel.FULL, fields=("template",)
    )
    assert route_a.content_signature(InChIKeyLevel.NO_STEREO, fields=("template",)) == route_b.content_signature(
        InChIKeyLevel.NO_STEREO, fields=("template",)
    )


@pytest.mark.unit
def test_content_signature_depth_limits_nested_reaction_content() -> None:
    child_a = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("C", KEY_A)], template="child-a"))
    child_b = molecule("CO", KEY_B, product_of=Reaction(reactants=[molecule("C", KEY_A)], template="child-b"))
    route_a = one_step_route([child_a, molecule("N", KEY_D)])
    route_b = one_step_route([child_b, molecule("N", KEY_D)])

    assert route_a.content_signature(fields=("template",), depth=1) == route_b.content_signature(
        fields=("template",), depth=1
    )
    assert route_a.content_signature(fields=("template",)) != route_b.content_signature(fields=("template",))


@pytest.mark.unit
def test_reaction_and_subtree_content_signatures_use_route_context() -> None:
    route = content_route(mapped_reaction_smiles="C.O>>CC")

    assert route.reaction_content_signatures(fields=("mapped_reaction_smiles",)) == {
        route.reaction_at("rc:r:/").content_signature(fields=("mapped_reaction_smiles",))
    }
    assert route.molecule_at("rc:m:/").content_subtree_signature(fields=("mapped_reaction_smiles",)) == (
        route.content_signature(fields=("mapped_reaction_smiles",))
    )


@pytest.mark.unit
def test_content_signature_rejects_unknown_reaction_fields() -> None:
    route = content_route()

    with pytest.raises(ValueError, match="unknown reaction content fields"):
        route.content_signature(fields=("condition_slot",))  # type: ignore[arg-type]


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
    assert [reactant.value.inchikey for reactant in permuted.reaction_at("rc:r:/").reactants()] == [
        reactant.value.inchikey for reactant in reference.reaction_at("rc:r:/").reactants()
    ]


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
