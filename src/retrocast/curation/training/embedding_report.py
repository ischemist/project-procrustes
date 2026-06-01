from __future__ import annotations

from collections.abc import Sequence

from retrocast.curation.training.embedding_audit import (
    CoverageSummary,
    InternalSubrouteEmbeddingSummary,
    QuerySetAudit,
    RouteEmbeddingAudit,
)
from retrocast.markdown import MarkdownAlign, MarkdownRow, markdown_table


def render_route_embedding_audit_markdown(audit: RouteEmbeddingAudit) -> str:
    lines = [
        f"# route embedding audit: {audit.release_name}",
        "",
        f"- match level: `{audit.match_level}`",
        f"- allow leaf extension: `{str(audit.allow_leaf_extension).lower()}`",
        "- partial minimum reactions: "
        f"{f'{audit.partial_min_reactions} reactions' if audit.partial_min_reactions is not None else 'not run'}",
        f"- training routes: {_format_integer(audit.training_routes)}",
        f"- training route signatures: {_format_integer(audit.training_route_signatures)}",
        f"- training reaction signatures: {_format_integer(audit.training_reaction_signatures)}",
        "",
        "terms: a query route is the route being searched for. "
        "a container route is a training route where an embedding is found.",
        "",
        _summary_table(audit.query_sets),
    ]

    for query_set in audit.query_sets:
        lines.extend(["", f"## {query_set.source}", ""])
        lines.extend(["### overlap summary", "", _overlap_summary(query_set)])
        lines.extend(["", _prefix_depth_summary(query_set)])
        if query_set.internal_subroute_embeddings is not None:
            lines.extend(["", _internal_subroute_summary(query_set, query_set.internal_subroute_embeddings)])
        lines.extend(["", _best_match_coverage(query_set.coverage)])
    return "\n".join(lines) + "\n"


def _summary_table(query_sets: Sequence[QuerySetAudit]) -> str:
    align: list[MarkdownAlign] = ["left"]
    for _ in query_sets:
        align.append("right")

    def row(label: str, values: Sequence[object]) -> MarkdownRow:
        return (label, *values)

    rows = [
        row("query routes", [_format_integer(q.query_routes) for q in query_sets]),
        row(
            "reaction signatures in training",
            [_format_count_rate(q.reaction_signature_overlap, q.query_reaction_signatures) for q in query_sets],
        ),
        row(
            "exact route signatures",
            [_format_count_rate(q.exact_route_signature_overlap, q.query_routes) for q in query_sets],
        ),
        row(
            "full route embeddings",
            [_format_count_rate(q.full_embeddings.query_routes_with_embedding, q.query_routes) for q in query_sets],
        ),
        row("routes with internal subroute embedding", [_format_internal_query_count(q) for q in query_sets]),
    ]
    return markdown_table(["metric", *(query_set.source for query_set in query_sets)], rows, align=align)


def _overlap_summary(query_set: QuerySetAudit) -> str:
    full = query_set.full_embeddings
    lines = [
        f"{_format_count_rate(query_set.reaction_signature_overlap, query_set.query_reaction_signatures)} "
        "reaction signatures are present in training; "
        f"{_format_count_rate(query_set.exact_route_signature_overlap, query_set.query_routes)} "
        "query routes have exact route-signature matches.",
        "",
    ]
    if full.query_routes_with_embedding == 0:
        lines.append("no query routes are fully embedded in training.")
        return "\n".join(lines)

    distance_counts = dict(full.root_distance_counts)
    lines.append(
        f"{_format_count_rate(full.query_routes_with_embedding, query_set.query_routes)} "
        "query routes are fully embedded somewhere inside training routes. "
        f"these produce {_format_integer(full.embedding_occurrences)} total matching occurrences. "
        f"{_format_plural(full.query_routes_with_root_shifted_embedding, 'query route has', 'query routes have')} "
        "a root-shifted full embedding; "
        f"{_format_plural(full.query_routes_with_leaf_extended_embedding, 'query route has', 'query routes have')} "
        "a leaf-extended full embedding. "
        f"{_format_plural(distance_counts.get(0, 0), 'occurrence shares', 'occurrences share')} "
        "the training target (distance 0 from the root); "
        f"{_format_root_distance_sentence(full.root_distance_counts)}."
    )
    return "\n".join(lines)


def _internal_subroute_summary(query_set: QuerySetAudit, summary: InternalSubrouteEmbeddingSummary) -> str:
    matches_per_embedded_subroute = (
        summary.embedding_occurrences / summary.embedded_internal_subroutes
        if summary.embedded_internal_subroutes
        else 0.0
    )
    return (
        f"{_format_count_rate(summary.query_routes_with_embedding, query_set.query_routes)} "
        f"query routes have at least one embedded internal subroute of {summary.min_reactions}+ reactions. "
        f"there are {_format_integer(summary.checked_internal_subroutes)} non-root internal subroutes "
        f"with at least {summary.min_reactions} reactions. "
        f"{_format_count_rate(summary.embedded_internal_subroutes, summary.checked_internal_subroutes)} "
        "of those internal subroutes are embedded. "
        f"there are {_format_integer(summary.embedding_occurrences)} total partial matching occurrences "
        f"({matches_per_embedded_subroute:.2f} "
        "matches per embedded internal subroute)."
    )


def _prefix_depth_summary(query_set: QuerySetAudit) -> str:
    return "\n".join(
        [
            "### root-prefix overlap",
            "",
            "query route prefixes are compared to training route prefixes with `route.signature(depth=k)`.",
            "",
            markdown_table(
                ["prefix depth", "query routes with depth", "found at training root", "found anywhere in training"],
                [
                    (
                        row.depth,
                        _format_integer(row.query_routes),
                        _format_count_rate(row.root_prefix_signature_overlap, row.query_routes),
                        _format_count_rate(row.subtree_prefix_signature_overlap, row.query_routes),
                    )
                    for row in query_set.prefix_depths
                ],
                align=["right", "right", "right", "right"],
            ),
        ]
    )


def _best_match_coverage(coverage: CoverageSummary) -> str:
    lines = [
        "### best-match coverage",
        "",
        "when overlap exists, how large and how specific is the largest overlap? "
        "each query route is compared to the single container route that embeds the largest part of it.",
        "",
        f"basis: {coverage.basis}.",
        "",
        "#### matched fraction of query route",
        "",
        "matched reactions divided by all reactions in the query route.",
        "",
        _coverage_table(
            all_query_routes=coverage.all_query_routes,
            embedded_query_routes=coverage.embedded_query_routes,
            all_values=coverage.all_query_fraction_stats,
            embedded_values=coverage.embedded_query_fraction_stats,
        ),
        "",
        "#### matched fraction of training route",
        "",
        "matched reactions divided by all reactions in the container route.",
        "",
        _coverage_table(
            all_query_routes=coverage.all_query_routes,
            embedded_query_routes=coverage.embedded_query_routes,
            all_values=coverage.all_container_fraction_stats,
            embedded_values=coverage.embedded_container_fraction_stats,
        ),
    ]
    if coverage.embedded_query_routes:
        lines.extend(
            [
                "",
                f"query routes with any embedding match "
                f"{coverage.embedded_mean_training_routes_per_query:.2f} "
                "unique training routes and "
                f"{coverage.embedded_mean_occurrences_per_query:.2f} "
                "total occurrences per query route on average.",
            ]
        )
    lines.extend(
        [
            "",
            "### largest embedded fraction of query-route reactions",
            "",
            "largest embedded match reactions divided by all reactions in the query route.",
            "",
            markdown_table(
                ["largest embedded fraction of query-route reactions", "query routes"],
                [
                    (bucket, _format_integer(query_routes))
                    for bucket, query_routes in coverage.matched_fraction_histogram
                ],
                align=["left", "right"],
            ),
        ]
    )
    return "\n".join(lines)


def _coverage_table(
    *,
    all_query_routes: int,
    embedded_query_routes: int,
    all_values: tuple[float, float, float],
    embedded_values: tuple[float, float, float],
) -> str:
    return markdown_table(
        ["population", "mean", "median", "p90"],
        [
            (
                f"all query routes ({_format_integer(all_query_routes)})",
                *(_format_percent(value) for value in all_values),
            ),
            (
                f"query routes with any embedding ({_format_integer(embedded_query_routes)})",
                *(_format_percent(value) for value in embedded_values),
            ),
        ],
        align=["left", "right", "right", "right"],
    )


def _format_count_rate(count: int, denominator: int) -> str:
    rate = count / denominator if denominator else 0.0
    return f"{_format_integer(count)} / {_format_integer(denominator)} ({_format_percent(rate)})"


def _format_internal_query_count(query_set: QuerySetAudit) -> str:
    if query_set.internal_subroute_embeddings is None:
        return "not run"
    return _format_count_rate(
        query_set.internal_subroute_embeddings.query_routes_with_embedding, query_set.query_routes
    )


def _format_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _format_integer(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _format_plural(value: int, singular: str, plural: str) -> str:
    return f"{_format_integer(value)} {singular if value == 1 else plural}"


def _format_root_distance_sentence(counts: Sequence[tuple[int, int]]) -> str:
    pieces = []
    for distance, occurrences in counts:
        if distance == 0:
            continue
        if distance == 1:
            description = "a root child"
        elif distance == 2:
            description = "a child of a root child"
        else:
            description = f"a depth-{distance} subtree"
        pieces.append(
            f"{_format_plural(occurrences, 'occurrence is', 'occurrences are')} embedded at distance "
            f"{distance} (a prefix of {description})"
        )
    if not pieces:
        return "no embeddings are below the target"
    return "; ".join(pieces)
