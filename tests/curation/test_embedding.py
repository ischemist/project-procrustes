from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrocast.chem import InChIKeyLevel
from retrocast.curation.embedding import find_route_embeddings, route_embeds_at, subtree_reaction_count
from retrocast.exceptions import InvalidRouteEmbeddingQueryError
from retrocast.models.route import Molecule, Reaction, Route, RoutePath
from retrocast.typing import InChIKeyStr, SmilesStr

PROPERTY_LABELS = tuple("abcdefgh")


def create_key(label: str, stereo_block: str = "UHFFFAOYSA") -> str:
    return f"{label.upper() * 14}-{stereo_block}-N"


LABEL_KEYS = {label: create_key(label) for label in (*PROPERTY_LABELS, "z")}
PROPERTY_KEYS = tuple(LABEL_KEYS[label] for label in PROPERTY_LABELS)
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H = PROPERTY_KEYS
KEY_Z = LABEL_KEYS["z"]
KEY_A_STEREO_1 = create_key("a", "BBBBBBBBSA")
KEY_A_STEREO_2 = create_key("a", "CCCCCCCCSA")


TreeShape = tuple[int, tuple["TreeShape", ...]]
LabelTree = tuple[str, tuple["LabelTree", ...]]


@dataclass(frozen=True, slots=True)
class AbstractEmbeddingCase:
    case_id: str
    query: LabelTree
    container: LabelTree
    embeds: bool
    blocked_without_leaf_extension: bool | None = None
    matched_reactions: int | None = None
    leaf_extensions: tuple[tuple[str, str], ...] = ()


def tree(label: str, *children: LabelTree) -> LabelTree:
    return (label, tuple(children))


@pytest.mark.unit
def test_exact_route_embeds_at_target_without_leaf_extension() -> None:
    query = linear_route()

    match = route_embeds_at(query.molecule_at(RoutePath.target()), query.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.query_path == RoutePath.target()
    assert match.container_path == RoutePath.target()
    assert match.matched_reactions == 2
    assert match.leaf_extensions == ()
    assert match.leaf_extended is False
    assert match.root_shifted is False
    assert subtree_reaction_count(query.molecule_at(RoutePath.target())) == 2


@pytest.mark.unit
def test_find_route_embeddings_reports_shifted_internal_subroute_match() -> None:
    query = Route(target=molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]))
    container = linear_route()

    matches = find_route_embeddings(query, container)

    assert [(match.container_path.id(), match.matched_reactions) for match in matches] == [("rc:m:/0", 1)]
    assert matches[0].root_shifted is True
    assert matches[0].leaf_extended is False


@pytest.mark.unit
def test_find_route_embeddings_rejects_molecule_only_query() -> None:
    query = Route(target=molecule("b", KEY_B))
    container = linear_route()

    with pytest.raises(InvalidRouteEmbeddingQueryError) as exc_info:
        find_route_embeddings(query, container)

    assert exc_info.value.code == "curation.route_embedding_query_invalid"
    assert exc_info.value.context == {
        "query_target_smiles": "b",
        "query_target_inchikey": KEY_B,
        "query_reactions": 0,
        "suggested_operation": "Route.contains_molecule",
    }


@pytest.mark.unit
def test_leaf_extension_records_query_boundary_and_counts_query_reactions_only() -> None:
    query = one_step_route(target_key=KEY_C, reactant_keys=[KEY_B])
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[molecule("b", KEY_B, reactants=[molecule("a", KEY_A)])],
        )
    )

    allowed = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    blocked = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert allowed is not None
    assert allowed.matched_reactions == 1
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in allowed.leaf_extensions] == [
        ("rc:m:/0", "rc:m:/0")
    ]
    assert allowed.leaf_extended is True
    assert blocked is None


@pytest.mark.unit
def test_non_leaf_query_does_not_embed_into_leaf_container() -> None:
    query = one_step_route(target_key=KEY_C, reactant_keys=[KEY_B])
    container = Route(target=molecule("c", KEY_C))

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is None


@pytest.mark.unit
def test_reaction_match_requires_the_same_reactant_multiset() -> None:
    query = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A, KEY_B])
    missing_duplicate = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A])
    wrong_multiplicity = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A, KEY_D])

    assert (
        route_embeds_at(query.molecule_at(RoutePath.target()), missing_duplicate.molecule_at(RoutePath.target()))
        is None
    )
    assert (
        route_embeds_at(query.molecule_at(RoutePath.target()), wrong_multiplicity.molecule_at(RoutePath.target()))
        is None
    )


@pytest.mark.unit
def test_same_key_sibling_matching_is_not_positional() -> None:
    query = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B),
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
            ],
        )
    )
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("b", KEY_B),
            ],
        )
    )

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.matched_reactions == 2
    assert match.leaf_extensions == ()


@pytest.mark.unit
def test_same_key_sibling_assignment_fails_when_no_complete_pairing_exists() -> None:
    query = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("b", KEY_B, reactants=[molecule("d", KEY_D)]),
            ],
        )
    )
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("b", KEY_B, reactants=[molecule("e", KEY_E)]),
            ],
        )
    )

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is None


@pytest.mark.unit
def test_match_level_controls_molecule_identity_for_embedding() -> None:
    query = Route(target=molecule("a", KEY_A_STEREO_1))
    container = Route(target=molecule("a", KEY_A_STEREO_2))

    full = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        InChIKeyLevel.FULL,
    )
    no_stereo = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        InChIKeyLevel.NO_STEREO,
    )

    assert full is None
    assert no_stereo is not None


@pytest.mark.unit
def test_find_route_embeddings_returns_all_matching_container_roots_in_traversal_order() -> None:
    query = one_step_route(target_key=KEY_B, reactant_keys=[KEY_A])
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("d", KEY_D),
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
            ],
        )
    )

    matches = find_route_embeddings(query, container)

    assert [match.container_path.id() for match in matches] == ["rc:m:/0", "rc:m:/2"]
    assert all(match.root_shifted for match in matches)


@pytest.mark.unit
@pytest.mark.regression
def test_regression_same_key_synthesized_siblings_backtrack_without_position_bias() -> None:
    query = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("b", KEY_B, reactants=[molecule("d", KEY_D)]),
            ],
        )
    )
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("d", KEY_D)]),
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
            ],
        )
    )

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.matched_reactions == 3
    assert match.leaf_extensions == ()


@pytest.mark.unit
@pytest.mark.regression
def test_regression_same_key_assignment_minimizes_leaf_extensions() -> None:
    query = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b0", KEY_B),
                molecule("b1", KEY_B),
                molecule("b2", KEY_B, reactants=[molecule("a", KEY_A)]),
            ],
        )
    )
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b0", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("b1", KEY_B),
                molecule("b2", KEY_B, reactants=[molecule("a", KEY_A)]),
            ],
        )
    )

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.matched_reactions == 2
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in match.leaf_extensions] == [
        ("rc:m:/0", "rc:m:/0")
    ]


@pytest.mark.unit
@pytest.mark.regression
def test_regression_zero_reactant_reaction_is_not_treated_as_a_leaf() -> None:
    query = Route(target=molecule("c", KEY_C, reactants=[]))
    container = Route(target=molecule("c", KEY_C, reactants=[]))
    leaf_query = Route(target=molecule("c", KEY_C))

    exact = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    allowed_leaf = route_embeds_at(
        leaf_query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
    )
    blocked_leaf = route_embeds_at(
        leaf_query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert exact is not None
    assert exact.matched_reactions == 1
    assert exact.leaf_extensions == ()
    assert allowed_leaf is not None
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in allowed_leaf.leaf_extensions] == [
        ("rc:m:/", "rc:m:/")
    ]
    assert blocked_leaf is None


@pytest.mark.unit
@pytest.mark.regression
def test_regression_shifted_match_preserves_nested_leaf_extension_paths() -> None:
    query = Route(target=molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]))
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule(
                    "b",
                    KEY_B,
                    reactants=[molecule("a", KEY_A, reactants=[molecule("z", KEY_Z)])],
                )
            ],
        )
    )

    matches = find_route_embeddings(query, container)

    assert len(matches) == 1
    assert matches[0].container_path.id() == "rc:m:/0"
    assert matches[0].root_shifted is True
    assert matches[0].matched_reactions == 1
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in matches[0].leaf_extensions] == [
        ("rc:m:/0", "rc:m:/0/0")
    ]


@pytest.mark.unit
@pytest.mark.regression
def test_regression_match_level_applies_to_reactants_inside_reaction_signatures() -> None:
    query = Route(target=molecule("c", KEY_C, reactants=[molecule("a", KEY_A_STEREO_1)]))
    container = Route(target=molecule("c", KEY_C, reactants=[molecule("a", KEY_A_STEREO_2)]))

    full = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        InChIKeyLevel.FULL,
    )
    no_stereo = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        InChIKeyLevel.NO_STEREO,
    )

    assert full is None
    assert no_stereo is not None


@pytest.mark.unit
@pytest.mark.regression
def test_regression_container_root_cannot_hide_extra_sibling_reactants() -> None:
    query = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A])
    container = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A, KEY_B])

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is None


@pytest.mark.unit
@pytest.mark.regression
def test_regression_root_molecule_identity_is_required_even_when_reaction_shape_matches() -> None:
    query = one_step_route(target_key=KEY_C, reactant_keys=[KEY_A])
    container = one_step_route(target_key=KEY_D, reactant_keys=[KEY_A])

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is None


@pytest.mark.unit
@pytest.mark.regression
def test_regression_route_embeds_at_supports_query_roots_below_query_target() -> None:
    query = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("d", KEY_D),
            ],
        )
    )
    container = Route(target=molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]))

    match = route_embeds_at(query.molecule_at("rc:m:/0"), container.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.query_path.id() == "rc:m:/0"
    assert match.container_path == RoutePath.target()
    assert match.matched_reactions == 1
    assert match.leaf_extensions == ()
    assert find_route_embeddings(query, container) == ()


@pytest.mark.unit
@pytest.mark.regression
def test_regression_non_convergent_query_embeds_in_convergent_container_root_by_leaf_extension() -> None:
    query = Route(target=molecule("c", KEY_C, reactants=[molecule("b", KEY_B), molecule("d", KEY_D)]))
    container = Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[
                molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]),
                molecule("d", KEY_D, reactants=[molecule("e", KEY_E)]),
            ],
        )
    )

    allowed = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    blocked = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert query.is_convergent() is False
    assert container.is_convergent() is True
    assert allowed is not None
    assert allowed.matched_reactions == 1
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in allowed.leaf_extensions] == [
        ("rc:m:/0", "rc:m:/0"),
        ("rc:m:/1", "rc:m:/1"),
    ]
    assert blocked is None


@pytest.mark.unit
@pytest.mark.regression
def test_regression_leaf_query_matches_upstream_expanded_container_root_only_when_allowed() -> None:
    query = Route(target=molecule("b", KEY_B))
    container = Route(
        target=molecule(
            "b",
            KEY_B,
            reactants=[molecule("a", KEY_A, reactants=[molecule("z", KEY_Z)])],
        )
    )

    allowed = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    blocked = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert allowed is not None
    assert allowed.matched_reactions == 0
    assert [(item.query_leaf_path.id(), item.container_path.id()) for item in allowed.leaf_extensions] == [
        ("rc:m:/", "rc:m:/")
    ]
    assert blocked is None


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize(
    "case",
    [
        AbstractEmbeddingCase(
            case_id="same_leaf",
            query=tree("a"),
            container=tree("a"),
            embeds=True,
            blocked_without_leaf_extension=True,
            matched_reactions=0,
        ),
        AbstractEmbeddingCase(
            case_id="different_leaf_label",
            query=tree("a"),
            container=tree("b"),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="same_one_step_tree",
            query=tree("a", tree("b")),
            container=tree("a", tree("b")),
            embeds=True,
            blocked_without_leaf_extension=True,
            matched_reactions=1,
        ),
        AbstractEmbeddingCase(
            case_id="query_deeper_than_container",
            query=tree("a", tree("b")),
            container=tree("a"),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="query_leaf_stops_at_expanded_container",
            query=tree("a"),
            container=tree("a", tree("b")),
            embeds=True,
            blocked_without_leaf_extension=False,
            matched_reactions=0,
            leaf_extensions=(("rc:m:/", "rc:m:/"),),
        ),
        AbstractEmbeddingCase(
            case_id="container_has_extra_sibling",
            query=tree("a", tree("b")),
            container=tree("a", tree("b"), tree("c")),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="container_missing_sibling",
            query=tree("a", tree("b"), tree("c")),
            container=tree("a", tree("b")),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="same_children_different_order",
            query=tree("a", tree("b"), tree("c")),
            container=tree("a", tree("c"), tree("b")),
            embeds=True,
            blocked_without_leaf_extension=True,
            matched_reactions=1,
        ),
        AbstractEmbeddingCase(
            case_id="duplicate_child_multiplicity_required",
            query=tree("a", tree("b"), tree("b")),
            container=tree("a", tree("b"), tree("c")),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="duplicate_children_pair_by_subtree_not_position",
            query=tree("a", tree("b", tree("c")), tree("b", tree("d"))),
            container=tree("a", tree("b", tree("d")), tree("b", tree("c"))),
            embeds=True,
            blocked_without_leaf_extension=True,
            matched_reactions=3,
        ),
        AbstractEmbeddingCase(
            case_id="duplicate_children_no_complete_pairing",
            query=tree("a", tree("b", tree("c")), tree("b", tree("d"))),
            container=tree("a", tree("b", tree("c")), tree("b", tree("e"))),
            embeds=False,
            blocked_without_leaf_extension=False,
        ),
        AbstractEmbeddingCase(
            case_id="branching_query_stops_at_deeper_branching_container",
            query=tree("a", tree("b"), tree("c")),
            container=tree("a", tree("c", tree("e")), tree("b", tree("d"))),
            embeds=True,
            blocked_without_leaf_extension=False,
            matched_reactions=1,
            leaf_extensions=(("rc:m:/0", "rc:m:/1"), ("rc:m:/1", "rc:m:/0")),
        ),
    ],
    ids=lambda case: case.case_id,
)
def test_abstract_rooted_tree_embedding_cases(case: AbstractEmbeddingCase) -> None:
    query = route_from_label_tree(case.query)
    container = route_from_label_tree(case.container)

    allowed = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    blocked = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert (allowed is not None) is case.embeds
    if case.blocked_without_leaf_extension is not None:
        assert (blocked is not None) is case.blocked_without_leaf_extension
    if allowed is not None:
        assert allowed.matched_reactions == case.matched_reactions
        assert sorted(
            (item.query_leaf_path.id(), item.container_path.id()) for item in allowed.leaf_extensions
        ) == sorted(case.leaf_extensions)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize(
    ("query", "container", "expected_container_paths"),
    [
        (tree("b", tree("c")), tree("a", tree("b", tree("c", tree("d")))), ("rc:m:/0",)),
        (tree("b", tree("c")), tree("a", tree("b", tree("c")), tree("d")), ("rc:m:/0",)),
        (tree("a", tree("b"), tree("c")), tree("a", tree("b", tree("d")), tree("c")), ("rc:m:/",)),
        (tree("a", tree("b"), tree("c")), tree("a", tree("b"), tree("c"), tree("d")), ()),
    ],
    ids=[
        "full_query_matches_internal_container_root",
        "one_step_query_matches_leaf_extended_internal_root",
        "branching_query_matches_expanded_branching_root",
        "full_query_rejects_container_with_extra_sibling",
    ],
)
def test_abstract_find_embeddings_scans_container_roots(
    query: LabelTree,
    container: LabelTree,
    expected_container_paths: tuple[str, ...],
) -> None:
    matches = find_route_embeddings(route_from_label_tree(query), route_from_label_tree(container))

    assert tuple(match.container_path.id() for match in matches) == expected_container_paths


def linear_route() -> Route:
    target = molecule("c", KEY_C, reactants=[molecule("b", KEY_B, reactants=[molecule("a", KEY_A)])])
    return Route(target=target)


def one_step_route(*, target_key: str, reactant_keys: list[str]) -> Route:
    return Route(
        target=molecule(
            "target", target_key, reactants=[molecule(f"r{index}", key) for index, key in enumerate(reactant_keys)]
        )
    )


def molecule(smiles: str, inchikey: str, *, reactants: list[Molecule] | None = None) -> Molecule:
    reaction = None if reactants is None else Reaction(reactants=reactants)
    return Molecule(smiles=SmilesStr(smiles), inchikey=InChIKeyStr(inchikey), product_of=reaction)


def route_from_label_tree(root: LabelTree) -> Route:
    def build(node: LabelTree, path: tuple[int, ...]) -> Molecule:
        label, children = node
        reactants = [build(child, (*path, index)) for index, child in enumerate(children)]
        return molecule("m" + "_".join(str(index) for index in path), LABEL_KEYS[label], reactants=reactants or None)

    return Route(target=build(root, ()))


@st.composite
def route_shapes(draw: st.DrawFn) -> TreeShape:
    leaf = st.integers(min_value=0, max_value=len(PROPERTY_KEYS) - 1).map(lambda label: (label, ()))
    subtree = st.recursive(
        leaf,
        lambda inner: st.tuples(
            st.integers(min_value=0, max_value=len(PROPERTY_KEYS) - 1),
            st.lists(inner, min_size=1, max_size=3).map(tuple),
        ),
        max_leaves=8,
    )
    return draw(subtree)


@st.composite
def branching_route_shapes(draw: st.DrawFn) -> TreeShape:
    children = draw(st.lists(route_shapes(), min_size=2, max_size=4))
    root_label = draw(st.integers(min_value=0, max_value=len(PROPERTY_KEYS) - 1))
    return (root_label, tuple(children))


def route_from_shape(shape: TreeShape) -> Route:
    def build(node: TreeShape, path: tuple[int, ...]) -> Molecule:
        label, children = node
        key = PROPERTY_KEYS[label]
        smiles = "m" + "_".join(str(index) for index in path)
        reactants = [build(child, (*path, index)) for index, child in enumerate(children)]
        return molecule(smiles, key, reactants=reactants or None)

    return Route(target=build(shape, ()))


def deepen_leaves(route: Route) -> Route:
    def copy_with_deeper_leaves(source: Molecule) -> Molecule:
        if source.product_of is None:
            return molecule(
                str(source.smiles),
                str(source.inchikey),
                reactants=[molecule(f"{source.smiles}_extension", KEY_Z)],
            )
        return molecule(
            str(source.smiles),
            str(source.inchikey),
            reactants=[copy_with_deeper_leaves(reactant) for reactant in source.product_of.reactants],
        )

    return Route(target=copy_with_deeper_leaves(route.target))


@pytest.mark.unit
@given(shape=route_shapes())
@settings(max_examples=75)
def test_generated_routes_embed_into_themselves(shape: TreeShape) -> None:
    route = route_from_shape(shape)

    match = route_embeds_at(route.molecule_at(RoutePath.target()), route.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.container_path == RoutePath.target()
    assert match.matched_reactions == subtree_reaction_count(route.molecule_at(RoutePath.target()))
    assert match.leaf_extensions == ()


@pytest.mark.unit
@given(shape=branching_route_shapes(), data=st.data())
@settings(max_examples=75)
def test_generated_sibling_order_does_not_change_embedding(shape: TreeShape, data: st.DataObject) -> None:
    child_count = len(shape[1])
    permutation = data.draw(st.permutations(range(child_count)))
    permuted = (shape[0], tuple(shape[1][index] for index in permutation))
    query = route_from_shape(shape)
    container = route_from_shape(permuted)

    match = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))

    assert match is not None
    assert match.matched_reactions == subtree_reaction_count(query.molecule_at(RoutePath.target()))
    assert match.leaf_extensions == ()


@pytest.mark.unit
@given(shape=route_shapes())
@settings(max_examples=75)
def test_generated_leaf_deepening_is_reported_only_as_leaf_extension(shape: TreeShape) -> None:
    query = route_from_shape(shape)
    container = deepen_leaves(query)

    allowed = route_embeds_at(query.molecule_at(RoutePath.target()), container.molecule_at(RoutePath.target()))
    blocked = route_embeds_at(
        query.molecule_at(RoutePath.target()),
        container.molecule_at(RoutePath.target()),
        allow_leaf_extension=False,
    )

    assert allowed is not None
    assert allowed.matched_reactions == subtree_reaction_count(query.molecule_at(RoutePath.target()))
    assert sorted(extension.query_leaf_path.id() for extension in allowed.leaf_extensions) == sorted(
        leaf.id() for leaf in query.leaves()
    )
    assert sorted(extension.container_path.id() for extension in allowed.leaf_extensions) == sorted(
        leaf.id() for leaf in query.leaves()
    )
    assert blocked is None


@pytest.mark.unit
@given(query_shape=route_shapes(), container_shape=route_shapes())
@settings(max_examples=75)
def test_find_route_embeddings_agrees_with_point_checks(query_shape: TreeShape, container_shape: TreeShape) -> None:
    query = route_from_shape(query_shape)
    container = route_from_shape(container_shape)
    query_root = query.molecule_at(RoutePath.target())

    if query_root.produced_by() is None:
        with pytest.raises(InvalidRouteEmbeddingQueryError):
            find_route_embeddings(query, container)
        return

    point_matches = tuple(
        match
        for container_root in container.iter_molecules()
        if (match := route_embeds_at(query_root, container_root)) is not None
    )

    assert find_route_embeddings(query, container) == point_matches
