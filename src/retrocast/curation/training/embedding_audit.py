from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal

from pydantic import BaseModel, ConfigDict

from retrocast.chem import InChIKeyLevel
from retrocast.curation.embedding import route_embeds_at, subtree_reaction_count
from retrocast.curation.training.records import TrainingRouteRecord
from retrocast.models.route import MoleculeView, Route, RoutePath

EmbeddingMatchKind = Literal["full_route", "internal_subroute"]


@dataclass(slots=True)
class ContainerRouteOccurrence:
    record: TrainingRouteRecord
    molecule: MoleculeView


@dataclass(slots=True)
class ContainerEmbeddingIndex:
    route_signatures: set[str]
    reaction_signatures: set[str]
    route_signature_counts: Mapping[str, int]
    reaction_signature_route_counts: Mapping[str, int]
    root_prefix_signatures_by_depth: Mapping[int, set[str]]
    subtree_prefix_signatures_by_depth: Mapping[int, set[str]]
    root_prefix_signature_counts_by_depth: Mapping[int, Mapping[str, int]]
    subtree_prefix_signature_counts_by_depth: Mapping[int, Mapping[str, int]]
    by_reaction_root: Mapping[str, tuple[ContainerRouteOccurrence, ...]]


class RouteEmbeddingLedgerRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_source: str
    query_id: str
    query_path: str
    query_route_reactions: int
    query_subroute_reactions: int
    container_route_id: str
    container_split: str
    container_path: str
    container_route_reactions: int
    container_subtree_reactions: int
    match_kind: EmbeddingMatchKind
    matched_reactions: int
    leaf_extension_query_paths: tuple[str, ...]
    leaf_extension_container_paths: tuple[str, ...]


@dataclass(slots=True)
class FullEmbeddingSummary:
    query_routes_with_embedding: int
    embedding_occurrences: int
    query_routes_with_root_shifted_embedding: int
    query_routes_with_leaf_extended_embedding: int
    root_distance_counts: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class InternalSubrouteEmbeddingSummary:
    min_reactions: int
    checked_internal_subroutes: int
    embedded_internal_subroutes: int
    query_routes_with_embedding: int
    embedding_occurrences: int


@dataclass(slots=True)
class PrefixDepthSummary:
    depth: int
    query_routes: int
    root_prefix_signature_overlap: int
    subtree_prefix_signature_overlap: int


@dataclass(slots=True)
class CoverageSummary:
    basis: str
    all_query_routes: int
    embedded_query_routes: int
    all_query_fraction_stats: tuple[float, float, float]
    embedded_query_fraction_stats: tuple[float, float, float]
    all_container_fraction_stats: tuple[float, float, float]
    embedded_container_fraction_stats: tuple[float, float, float]
    embedded_mean_container_routes_per_query: float
    embedded_mean_occurrences_per_query: float
    matched_fraction_histogram: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class QuerySetAudit:
    source: str
    query_routes: int
    query_reaction_signatures: int
    exact_route_signature_overlap: int
    reaction_signature_overlap: int
    prefix_depths: tuple[PrefixDepthSummary, ...]
    full_embeddings: FullEmbeddingSummary
    internal_subroute_embeddings: InternalSubrouteEmbeddingSummary | None
    coverage: CoverageSummary


@dataclass(slots=True)
class RouteEmbeddingAudit:
    release_name: str
    match_level: str
    allow_leaf_extension: bool
    partial_min_reactions: int | None
    container_routes: int
    container_route_signatures: int
    container_reaction_signatures: int
    query_sets: tuple[QuerySetAudit, ...]
    ledger_rows: tuple[RouteEmbeddingLedgerRow, ...]


@dataclass(slots=True)
class _QueryCoverageRow:
    matched_fraction_of_query_route: float
    matched_fraction_of_container_route: float
    container_routes: int
    occurrences: int


def build_route_embedding_audit(
    *,
    release_name: str,
    container_records: Sequence[TrainingRouteRecord],
    queries_by_source: Mapping[str, Mapping[str, Route]],
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    allow_leaf_extension: bool = True,
    include_partial: bool = False,
    partial_min_reactions: int = 2,
    exclude_query_containers: bool = False,
) -> RouteEmbeddingAudit:
    if exclude_query_containers:
        container_ids = [record.id for record in container_records]
        duplicate_ids = {container_id for container_id, count in Counter(container_ids).items() if count > 1}
        if duplicate_ids:
            raise ValueError(
                f"exclude_query_containers requires unique container record ids; duplicate ids: {sorted(duplicate_ids)}"
            )

        container_id_set = set(container_ids)
        query_ids = {query_id for queries in queries_by_source.values() for query_id in queries}
        missing_query_ids = sorted(query_ids - container_id_set)
        if missing_query_ids:
            raise ValueError(
                "exclude_query_containers requires query ids to match container record ids; "
                f"missing ids: {missing_query_ids}"
            )
    index = build_container_embedding_index(container_records, match_level)
    query_sets: list[QuerySetAudit] = []
    ledger_rows: list[RouteEmbeddingLedgerRow] = []
    for source, queries in sorted(queries_by_source.items()):
        query_set, rows = _audit_query_set(
            source=source,
            queries=queries,
            index=index,
            match_level=match_level,
            allow_leaf_extension=allow_leaf_extension,
            include_partial=include_partial,
            partial_min_reactions=partial_min_reactions,
            exclude_query_containers=exclude_query_containers,
        )
        query_sets.append(query_set)
        ledger_rows.extend(rows)

    return RouteEmbeddingAudit(
        release_name=release_name,
        match_level=match_level.value,
        allow_leaf_extension=allow_leaf_extension,
        partial_min_reactions=partial_min_reactions if include_partial else None,
        container_routes=len(container_records),
        container_route_signatures=len(index.route_signatures),
        container_reaction_signatures=len(index.reaction_signatures),
        query_sets=tuple(query_sets),
        ledger_rows=tuple(ledger_rows),
    )


def build_container_embedding_index(
    records: Sequence[TrainingRouteRecord],
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> ContainerEmbeddingIndex:
    route_signatures: set[str] = set()
    reaction_signatures: set[str] = set()
    route_signature_counts: Counter[str] = Counter()
    reaction_signature_route_counts: Counter[str] = Counter()
    root_prefix_signatures_by_depth: dict[int, set[str]] = defaultdict(set)
    subtree_prefix_signatures_by_depth: dict[int, set[str]] = defaultdict(set)
    root_prefix_signature_counts_by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    subtree_prefix_signature_counts_by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    by_reaction_root: dict[str, list[ContainerRouteOccurrence]] = defaultdict(list)

    for record in records:
        route_signature = record.route.signature(match_level)
        route_signatures.add(route_signature)
        route_signature_counts[route_signature] += 1
        record_reaction_signatures = set(record.route.reaction_signatures(match_level))
        reaction_signatures.update(record_reaction_signatures)
        reaction_signature_route_counts.update(record_reaction_signatures)
        for depth in range(1, record.route.depth() + 1):
            root_prefix_signature = record.route.signature(match_level, depth=depth)
            root_prefix_signatures_by_depth[depth].add(root_prefix_signature)
            root_prefix_signature_counts_by_depth[depth][root_prefix_signature] += 1
        subtree_prefixes_by_depth: dict[int, set[str]] = defaultdict(set)
        for molecule in record.route.iter_molecules():
            for depth in range(1, molecule.depth() + 1):
                subtree_prefix = molecule.subtree_signature(match_level, depth=depth)
                subtree_prefix_signatures_by_depth[depth].add(subtree_prefix)
                subtree_prefixes_by_depth[depth].add(subtree_prefix)
            occurrence = ContainerRouteOccurrence(record=record, molecule=molecule)
            reaction = molecule.produced_by()
            if reaction is not None:
                by_reaction_root[reaction.signature(match_level)].append(occurrence)
        for depth, signatures in subtree_prefixes_by_depth.items():
            subtree_prefix_signature_counts_by_depth[depth].update(signatures)

    return ContainerEmbeddingIndex(
        route_signatures=route_signatures,
        reaction_signatures=reaction_signatures,
        route_signature_counts=dict(route_signature_counts),
        reaction_signature_route_counts=dict(reaction_signature_route_counts),
        root_prefix_signatures_by_depth=dict(root_prefix_signatures_by_depth),
        subtree_prefix_signatures_by_depth=dict(subtree_prefix_signatures_by_depth),
        root_prefix_signature_counts_by_depth={
            depth: dict(counts) for depth, counts in root_prefix_signature_counts_by_depth.items()
        },
        subtree_prefix_signature_counts_by_depth={
            depth: dict(counts) for depth, counts in subtree_prefix_signature_counts_by_depth.items()
        },
        by_reaction_root={key: tuple(value) for key, value in by_reaction_root.items()},
    )


def _audit_query_set(
    *,
    source: str,
    queries: Mapping[str, Route],
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
    include_partial: bool,
    partial_min_reactions: int,
    exclude_query_containers: bool,
) -> tuple[QuerySetAudit, tuple[RouteEmbeddingLedgerRow, ...]]:
    full_rows = _full_route_ledger_rows(
        source=source,
        queries=queries,
        index=index,
        match_level=match_level,
        allow_leaf_extension=allow_leaf_extension,
        exclude_query_containers=exclude_query_containers,
    )
    internal_summary = None
    internal_rows: tuple[RouteEmbeddingLedgerRow, ...] = ()
    if include_partial:
        internal_summary, internal_rows = _internal_subroute_audit(
            source=source,
            queries=queries,
            index=index,
            match_level=match_level,
            allow_leaf_extension=allow_leaf_extension,
            partial_min_reactions=partial_min_reactions,
            exclude_query_containers=exclude_query_containers,
        )

    rows = (*full_rows, *internal_rows)
    query_ids = list(queries)
    query_reaction_signatures = {
        signature for route in queries.values() for signature in route.reaction_signatures(match_level)
    }
    if exclude_query_containers:
        exact_route_signature_overlap = sum(
            index.route_signature_counts.get(route.signature(match_level), 0) > 1 for route in queries.values()
        )
        reaction_signature_overlap = len(
            {
                signature
                for signature in query_reaction_signatures
                if index.reaction_signature_route_counts.get(signature, 0) > 1
            }
        )
    else:
        exact_route_signature_overlap = sum(
            route.signature(match_level) in index.route_signatures for route in queries.values()
        )
        reaction_signature_overlap = len(query_reaction_signatures & index.reaction_signatures)
    query_set = QuerySetAudit(
        source=source,
        query_routes=len(queries),
        query_reaction_signatures=len(query_reaction_signatures),
        exact_route_signature_overlap=exact_route_signature_overlap,
        reaction_signature_overlap=reaction_signature_overlap,
        prefix_depths=_summarize_prefix_depths(
            queries=queries,
            index=index,
            match_level=match_level,
            exclude_query_containers=exclude_query_containers,
        ),
        full_embeddings=_summarize_full_embeddings(full_rows),
        internal_subroute_embeddings=internal_summary,
        coverage=_summarize_coverage(
            query_ids=query_ids,
            rows=rows,
            basis=(
                f"full routes + internal subroutes with {partial_min_reactions}+ reactions"
                if include_partial
                else "full routes only"
            ),
        ),
    )
    return query_set, rows


def _summarize_prefix_depths(
    *,
    queries: Mapping[str, Route],
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
    exclude_query_containers: bool,
) -> tuple[PrefixDepthSummary, ...]:
    max_depth = max((route.depth() for route in queries.values()), default=0)
    summaries: list[PrefixDepthSummary] = []
    for depth in range(1, max_depth + 1):
        eligible = [route for route in queries.values() if route.depth() >= depth]
        root_prefixes = index.root_prefix_signatures_by_depth.get(depth, set())
        subtree_prefixes = index.subtree_prefix_signatures_by_depth.get(depth, set())
        root_counts = index.root_prefix_signature_counts_by_depth.get(depth, {})
        subtree_counts = index.subtree_prefix_signature_counts_by_depth.get(depth, {})
        signatures = [route.signature(match_level, depth=depth) for route in eligible]
        if exclude_query_containers:
            root_overlap = sum(root_counts.get(signature, 0) > 1 for signature in signatures)
            subtree_overlap = sum(subtree_counts.get(signature, 0) > 1 for signature in signatures)
        else:
            root_overlap = sum(signature in root_prefixes for signature in signatures)
            subtree_overlap = sum(signature in subtree_prefixes for signature in signatures)
        summaries.append(
            PrefixDepthSummary(
                depth=depth,
                query_routes=len(eligible),
                root_prefix_signature_overlap=root_overlap,
                subtree_prefix_signature_overlap=subtree_overlap,
            )
        )
    return tuple(summaries)


def _full_route_ledger_rows(
    *,
    source: str,
    queries: Mapping[str, Route],
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
    exclude_query_containers: bool,
) -> tuple[RouteEmbeddingLedgerRow, ...]:
    rows: list[RouteEmbeddingLedgerRow] = []
    for query_id, route in queries.items():
        rows.extend(
            _embedding_rows(
                query_source=source,
                query_id=query_id,
                query_route=route,
                query_molecule=route.molecule_at(RoutePath.target()),
                match_kind="full_route",
                index=index,
                match_level=match_level,
                allow_leaf_extension=allow_leaf_extension,
                exclude_query_container=exclude_query_containers,
            )
        )
    return tuple(rows)


def _internal_subroute_audit(
    *,
    source: str,
    queries: Mapping[str, Route],
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
    partial_min_reactions: int,
    exclude_query_containers: bool,
) -> tuple[InternalSubrouteEmbeddingSummary, tuple[RouteEmbeddingLedgerRow, ...]]:
    checked = 0
    rows: list[RouteEmbeddingLedgerRow] = []
    embedded_subroutes: set[tuple[str, str]] = set()
    for query_id, route in queries.items():
        for molecule in route.iter_molecules():
            if molecule.path == RoutePath.target() or subtree_reaction_count(molecule) < partial_min_reactions:
                continue
            checked += 1
            subroute_key = (query_id, molecule.path.id())
            subroute_rows = _embedding_rows(
                query_source=source,
                query_id=query_id,
                query_route=route,
                query_molecule=molecule,
                match_kind="internal_subroute",
                index=index,
                match_level=match_level,
                allow_leaf_extension=allow_leaf_extension,
                exclude_query_container=exclude_query_containers,
            )
            rows.extend(subroute_rows)
            if subroute_rows:
                embedded_subroutes.add(subroute_key)

    return (
        InternalSubrouteEmbeddingSummary(
            min_reactions=partial_min_reactions,
            checked_internal_subroutes=checked,
            embedded_internal_subroutes=len(embedded_subroutes),
            query_routes_with_embedding=len({query_id for query_id, _ in embedded_subroutes}),
            embedding_occurrences=len(rows),
        ),
        tuple(rows),
    )


def _embedding_rows(
    *,
    query_source: str,
    query_id: str,
    query_route: Route,
    query_molecule: MoleculeView,
    match_kind: EmbeddingMatchKind,
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
    exclude_query_container: bool,
) -> tuple[RouteEmbeddingLedgerRow, ...]:
    rows: list[RouteEmbeddingLedgerRow] = []
    query_route_reactions = sum(1 for _ in query_route.iter_reactions())
    query_subroute_reactions = subtree_reaction_count(query_molecule)
    for occurrence in _candidate_occurrences(query_molecule, index, match_level):
        if exclude_query_container and occurrence.record.id == query_id:
            continue
        match = route_embeds_at(
            query_molecule,
            occurrence.molecule,
            match_level,
            allow_leaf_extension=allow_leaf_extension,
        )
        if match is None:
            continue

        container_route_reactions = sum(1 for _ in occurrence.record.route.iter_reactions())
        container_subtree_reactions = subtree_reaction_count(occurrence.molecule)
        rows.append(
            RouteEmbeddingLedgerRow(
                query_source=query_source,
                query_id=query_id,
                query_path=match.query_path.id(),
                query_route_reactions=query_route_reactions,
                query_subroute_reactions=query_subroute_reactions,
                container_route_id=occurrence.record.id,
                container_split=occurrence.record.split,
                container_path=match.container_path.id(),
                container_route_reactions=container_route_reactions,
                container_subtree_reactions=container_subtree_reactions,
                match_kind=match_kind,
                matched_reactions=match.matched_reactions,
                leaf_extension_query_paths=tuple(extension.query_leaf_path.id() for extension in match.leaf_extensions),
                leaf_extension_container_paths=tuple(
                    extension.container_path.id() for extension in match.leaf_extensions
                ),
            )
        )
    return tuple(rows)


def _candidate_occurrences(
    query_root: MoleculeView,
    index: ContainerEmbeddingIndex,
    match_level: InChIKeyLevel,
) -> Sequence[ContainerRouteOccurrence]:
    reaction = query_root.produced_by()
    if reaction is None:
        return ()
    return index.by_reaction_root.get(reaction.signature(match_level), ())


def _summarize_full_embeddings(rows: Sequence[RouteEmbeddingLedgerRow]) -> FullEmbeddingSummary:
    root_distance_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        root_distance_counts[RoutePath.parse(row.container_path).depth()] += 1

    return FullEmbeddingSummary(
        query_routes_with_embedding=len({row.query_id for row in rows}),
        embedding_occurrences=len(rows),
        query_routes_with_root_shifted_embedding=len(
            {row.query_id for row in rows if row.container_path != RoutePath.target().id()}
        ),
        query_routes_with_leaf_extended_embedding=len({row.query_id for row in rows if row.leaf_extension_query_paths}),
        root_distance_counts=tuple(
            (distance, root_distance_counts[distance]) for distance in sorted(root_distance_counts)
        ),
    )


def _summarize_coverage(
    *,
    query_ids: Sequence[str],
    rows: Sequence[RouteEmbeddingLedgerRow],
    basis: str,
) -> CoverageSummary:
    rows_by_query: dict[str, list[RouteEmbeddingLedgerRow]] = defaultdict(list)
    for row in rows:
        rows_by_query[row.query_id].append(row)

    coverage_rows = [_query_coverage_row(rows_by_query.get(query_id, ())) for query_id in query_ids]
    embedded_rows = [row for row in coverage_rows if row.occurrences]
    return CoverageSummary(
        basis=basis,
        all_query_routes=len(coverage_rows),
        embedded_query_routes=len(embedded_rows),
        all_query_fraction_stats=_fraction_stats([row.matched_fraction_of_query_route for row in coverage_rows]),
        embedded_query_fraction_stats=_fraction_stats([row.matched_fraction_of_query_route for row in embedded_rows]),
        all_container_fraction_stats=_fraction_stats(
            [row.matched_fraction_of_container_route for row in coverage_rows]
        ),
        embedded_container_fraction_stats=_fraction_stats(
            [row.matched_fraction_of_container_route for row in embedded_rows]
        ),
        embedded_mean_container_routes_per_query=_mean([row.container_routes for row in embedded_rows]),
        embedded_mean_occurrences_per_query=_mean([row.occurrences for row in embedded_rows]),
        matched_fraction_histogram=_coverage_histogram(coverage_rows),
    )


def _query_coverage_row(rows: Sequence[RouteEmbeddingLedgerRow]) -> _QueryCoverageRow:
    if not rows:
        return _QueryCoverageRow(
            matched_fraction_of_query_route=0.0,
            matched_fraction_of_container_route=0.0,
            container_routes=0,
            occurrences=0,
        )

    best = max(
        rows,
        key=lambda row: (
            _ratio(row.matched_reactions, row.query_route_reactions),
            _ratio(row.matched_reactions, row.container_subtree_reactions),
            _ratio(row.matched_reactions, row.container_route_reactions),
            row.matched_reactions,
        ),
    )
    return _QueryCoverageRow(
        matched_fraction_of_query_route=_ratio(best.matched_reactions, best.query_route_reactions),
        matched_fraction_of_container_route=_ratio(best.matched_reactions, best.container_route_reactions),
        container_routes=len({row.container_route_id for row in rows}),
        occurrences=len(rows),
    )


def _coverage_histogram(rows: Sequence[_QueryCoverageRow]) -> tuple[tuple[str, int], ...]:
    buckets = ("0%", "(0,25%]", "(25,50%]", "(50,75%]", "(75,100%)", "100%")
    counts = dict.fromkeys(buckets, 0)
    for row in rows:
        counts[_coverage_bucket(row.matched_fraction_of_query_route)] += 1
    return tuple((bucket, counts[bucket]) for bucket in buckets)


def _coverage_bucket(value: float) -> str:
    if value == 0:
        return "0%"
    if value <= 0.25:
        return "(0,25%]"
    if value <= 0.50:
        return "(25,50%]"
    if value <= 0.75:
        return "(50,75%]"
    if value < 1.0:
        return "(75,100%)"
    return "100%"


def _mean(values: Sequence[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fraction_stats(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[midpoint]
    else:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
    return (_mean(values), median, sorted_values[ceil(0.9 * len(sorted_values)) - 1])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
