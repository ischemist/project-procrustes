from __future__ import annotations

from retrocast.chem import get_inchi_key
from retrocast.curation.training import TrainingRouteRecord
from retrocast.curation.training.embedding_audit import (
    QueryRoute,
    benchmark_query_routes,
    build_route_embedding_audit,
)
from retrocast.curation.training.embedding_report import render_route_embedding_audit_markdown
from retrocast.exceptions import InvalidRouteEmbeddingQueryError
from retrocast.models.route import Molecule, Reaction, Route
from retrocast.models.task import Benchmark, Target
from retrocast.typing import InChIKeyStr, SmilesStr

KEY_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
KEY_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"
KEY_C = "CCCCCCCCCCCCCC-UHFFFAOYSA-N"
KEY_D = "DDDDDDDDDDDDDD-UHFFFAOYSA-N"


def test_route_embedding_audit_summarizes_full_matches_and_renders_markdown() -> None:
    training = route_record("container", route_c_b_a())
    queries = [
        QueryRoute(
            source="bench", id="shifted", route=Route(target=molecule("b", KEY_B, reactants=[molecule("a", KEY_A)]))
        ),
        QueryRoute(
            source="bench", id="leaf", route=Route(target=molecule("c", KEY_C, reactants=[molecule("b", KEY_B)]))
        ),
        QueryRoute(
            source="bench", id="absent", route=Route(target=molecule("d", KEY_D, reactants=[molecule("a", KEY_A)]))
        ),
    ]

    audit = build_route_embedding_audit(
        release_name="route-holdout-n1-n5",
        training_records=[training],
        queries_by_source={"bench": queries},
    )

    query_set = audit.query_sets[0]
    full = query_set.full_embeddings
    assert query_set.query_routes == 3
    assert query_set.reaction_signature_overlap == 2
    assert query_set.exact_route_signature_overlap == 0
    assert full.query_routes_with_embedding == 2
    assert full.embedding_occurrences == 2
    assert full.query_routes_with_root_shifted_embedding == 1
    assert full.query_routes_with_leaf_extended_embedding == 1
    assert full.root_distance_counts == ((0, 1), (1, 1))
    assert [(row.query_id, row.container_path, row.leaf_extension_query_paths) for row in audit.ledger_rows] == [
        ("shifted", "rc:m:/0", ()),
        ("leaf", "rc:m:/", ("rc:m:/0",)),
    ]

    markdown = render_route_embedding_audit_markdown(audit)
    assert "# route embedding audit: route-holdout-n1-n5" in markdown
    assert "| full route embeddings | 2 / 3 (66.7%) |" in markdown
    assert "1 query route has a root-shifted full embedding" in markdown
    assert "1 query route has a leaf-extended full embedding" in markdown


def test_route_embedding_audit_can_include_internal_subroutes() -> None:
    training = route_record("container", route_c_b_a())
    query = QueryRoute(source="bench", id="full", route=route_c_b_a())

    audit = build_route_embedding_audit(
        release_name="route-holdout-n1-n5",
        training_records=[training],
        queries_by_source={"bench": [query]},
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


def test_route_embedding_audit_rejects_molecule_only_queries() -> None:
    query = QueryRoute(source="bench", id="molecule", route=Route(target=molecule("a", KEY_A)))

    try:
        build_route_embedding_audit(
            release_name="route-holdout-n1-n5",
            training_records=[route_record("container", route_c_b_a())],
            queries_by_source={"bench": [query]},
        )
    except InvalidRouteEmbeddingQueryError as exc:
        assert exc.context["query_source"] == "bench"
        assert exc.context["query_id"] == "molecule"
    else:
        raise AssertionError("molecule-only query should fail")


def test_benchmark_query_routes_selects_primary_or_all_acceptable_routes() -> None:
    benchmark = Benchmark(
        name="bench",
        targets={
            "target": Target(
                id="target",
                smiles=SmilesStr("C"),
                inchikey=InChIKeyStr(get_inchi_key("C")),
                acceptable_routes=[route_c_b_a(), route_c_b_a()],
            )
        },
    )

    primary = benchmark_query_routes(benchmark, source="bench", route_selection="primary")
    all_routes = benchmark_query_routes(benchmark, source="bench", route_selection="all")

    assert [(query.source, query.id) for query in primary] == [("bench", "target")]
    assert [(query.source, query.id) for query in all_routes] == [("bench", "target:1"), ("bench", "target:2")]


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
