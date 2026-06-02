from __future__ import annotations

import pytest

from retrocast.curation.training import TrainingRouteRecord
from retrocast.curation.training.embedding_audit import RouteEmbeddingAudit, build_route_embedding_audit
from retrocast.curation.training.embedding_report import render_route_embedding_audit_markdown
from retrocast.models.route import Molecule, Reaction, Route
from retrocast.typing import InChIKeyStr, SmilesStr

KEY_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
KEY_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"
KEY_C = "CCCCCCCCCCCCCC-UHFFFAOYSA-N"
KEY_D = "DDDDDDDDDDDDDD-UHFFFAOYSA-N"


@pytest.mark.integration
def test_route_embedding_audit_summarizes_full_matches() -> None:
    audit = sample_full_match_audit()
    query_set = audit.query_sets[0]
    full = query_set.full_embeddings

    assert query_set.query_routes == 3
    assert query_set.reaction_signature_overlap == 2
    assert query_set.exact_route_signature_overlap == 0
    assert [
        (
            row.depth,
            row.query_routes,
            row.root_prefix_signature_overlap,
            row.subtree_prefix_signature_overlap,
        )
        for row in query_set.prefix_depths
    ] == [(1, 3, 1, 2)]
    assert full.query_routes_with_embedding == 2
    assert full.embedding_occurrences == 2
    assert full.query_routes_with_root_shifted_embedding == 1
    assert full.query_routes_with_leaf_extended_embedding == 1
    assert full.root_distance_counts == ((0, 1), (1, 1))
    assert [(row.query_id, row.container_path, row.leaf_extension_query_paths) for row in audit.ledger_rows] == [
        ("shifted", "rc:m:/0", ()),
        ("leaf", "rc:m:/", ("rc:m:/0",)),
    ]


@pytest.mark.integration
def test_route_embedding_report_renders_audit_summary() -> None:
    audit = sample_full_match_audit()

    markdown = render_route_embedding_audit_markdown(audit)

    assert "# route embedding audit: route-holdout-n1-n5" in markdown
    assert "| full route embeddings | 2 / 3 (66.7%) |" in markdown
    assert "### root-prefix overlap" in markdown
    assert "| 1 | 3 | 1 / 3 (33.3%) | 2 / 3 (66.7%) |" in markdown
    assert "1 query route has a root-shifted full embedding" in markdown
    assert "1 query route has a leaf-extended full embedding" in markdown


@pytest.mark.integration
def test_route_embedding_audit_can_include_internal_subroutes() -> None:
    training = route_record("container", route_c_b_a())

    audit = build_route_embedding_audit(
        release_name="route-holdout-n1-n5",
        training_records=[training],
        queries_by_source={"bench": {"full": route_c_b_a()}},
        include_partial=True,
        partial_min_reactions=1,
    )

    query_set = audit.query_sets[0]
    internal = query_set.internal_subroute_embeddings
    assert internal is not None
    assert internal.checked_internal_subroutes == 1
    assert internal.embedded_internal_subroutes == 1
    assert internal.query_routes_with_embedding == 1
    assert internal.embedding_occurrences == 1
    assert [row.match_kind for row in audit.ledger_rows] == ["full_route", "internal_subroute"]
    assert query_set.coverage.embedded_mean_occurrences_per_query == 2
    assert [
        (
            row.depth,
            row.query_routes,
            row.root_prefix_signature_overlap,
            row.subtree_prefix_signature_overlap,
        )
        for row in query_set.prefix_depths
    ] == [
        (1, 1, 1, 1),
        (2, 1, 1, 1),
    ]


@pytest.mark.integration
def test_route_embedding_audit_can_exclude_query_containers() -> None:
    route = route_c_b_a()
    record = route_record("self", route)

    audit = build_route_embedding_audit(
        release_name="benchmark-route-embeddings",
        training_records=[record],
        queries_by_source={"bench": {record.id: route}},
        include_partial=True,
        partial_min_reactions=1,
        exclude_query_containers=True,
    )

    query_set = audit.query_sets[0]
    internal = query_set.internal_subroute_embeddings
    assert internal is not None
    assert query_set.reaction_signature_overlap == 0
    assert query_set.exact_route_signature_overlap == 0
    assert [
        (
            row.depth,
            row.root_prefix_signature_overlap,
            row.subtree_prefix_signature_overlap,
        )
        for row in query_set.prefix_depths
    ] == [(1, 0, 0), (2, 0, 0)]
    assert query_set.full_embeddings.query_routes_with_embedding == 0
    assert internal.query_routes_with_embedding == 0
    assert audit.ledger_rows == ()


@pytest.mark.integration
def test_route_embedding_audit_rejects_exclusion_without_matching_query_id() -> None:
    with pytest.raises(ValueError, match="query ids to match TrainingRouteRecord.id"):
        build_route_embedding_audit(
            release_name="benchmark-route-embeddings",
            training_records=[route_record("container", route_c_b_a())],
            queries_by_source={"bench": {"query": route_c_b_a()}},
            exclude_query_containers=True,
        )


@pytest.mark.integration
def test_route_embedding_audit_rejects_exclusion_with_duplicate_container_ids() -> None:
    route = route_c_b_a()
    records = [route_record("duplicate", route), route_record("duplicate", route)]

    with pytest.raises(ValueError, match="unique TrainingRouteRecord.id"):
        build_route_embedding_audit(
            release_name="benchmark-route-embeddings",
            training_records=records,
            queries_by_source={"bench": {records[0].id: route}},
            exclude_query_containers=True,
        )


def sample_full_match_audit() -> RouteEmbeddingAudit:
    training = route_record("container", route_c_b_a())
    queries = {
        "shifted": Route(target=molecule("b", KEY_B, reactants=[molecule("a", KEY_A)])),
        "leaf": Route(target=molecule("c", KEY_C, reactants=[molecule("b", KEY_B)])),
        "absent": Route(target=molecule("d", KEY_D, reactants=[molecule("a", KEY_A)])),
    }

    return build_route_embedding_audit(
        release_name="route-holdout-n1-n5",
        training_records=[training],
        queries_by_source={"bench": queries},
    )


def route_c_b_a() -> Route:
    return Route(
        target=molecule(
            "c",
            KEY_C,
            reactants=[molecule("b", KEY_B, reactants=[molecule("a", KEY_A)])],
        )
    )


def molecule(label: str, key: str, *, reactants: list[Molecule] | None = None) -> Molecule:
    return Molecule(
        smiles=SmilesStr(label),
        inchikey=InChIKeyStr(key),
        product_of=Reaction(reactants=reactants) if reactants is not None else None,
    )


def route_record(name: str, route: Route) -> TrainingRouteRecord:
    return TrainingRouteRecord(id=f"route-{name}", split="training", route=route)
