from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from retrocast.chem import InChIKeyLevel
from retrocast.curation.training.records import TrainingRouteRecord
from retrocast.models.route import Route

EmbeddingMatchKind = Literal["full_route", "internal_subroute"]


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
    """Build a release embedding audit in the Rust core."""
    from retrocast import native

    payload = native.build_route_embedding_audit(
        release_name,
        container_records,
        queries_by_source,
        match_level=match_level,
        allow_leaf_extension=allow_leaf_extension,
        include_partial=include_partial,
        partial_min_reactions=partial_min_reactions,
        exclude_query_containers=exclude_query_containers,
    )
    return _audit_from_native(payload)


def _audit_from_native(payload: dict[str, Any]) -> RouteEmbeddingAudit:
    query_sets = []
    for raw_query_set in payload["query_sets"]:
        query_set = dict(raw_query_set)
        raw_full = dict(query_set["full_embeddings"])
        raw_full["root_distance_counts"] = tuple(tuple(row) for row in raw_full["root_distance_counts"])
        raw_internal = query_set["internal_subroute_embeddings"]
        raw_coverage = dict(query_set["coverage"])
        for field in (
            "all_query_fraction_stats",
            "embedded_query_fraction_stats",
            "all_container_fraction_stats",
            "embedded_container_fraction_stats",
            "matched_fraction_histogram",
        ):
            raw_coverage[field] = tuple(tuple(row) if isinstance(row, list) else row for row in raw_coverage[field])
        query_sets.append(
            QuerySetAudit(
                source=str(query_set["source"]),
                query_routes=int(query_set["query_routes"]),
                query_reaction_signatures=int(query_set["query_reaction_signatures"]),
                exact_route_signature_overlap=int(query_set["exact_route_signature_overlap"]),
                reaction_signature_overlap=int(query_set["reaction_signature_overlap"]),
                prefix_depths=tuple(PrefixDepthSummary(**dict(row)) for row in query_set["prefix_depths"]),
                full_embeddings=FullEmbeddingSummary(**raw_full),
                internal_subroute_embeddings=None
                if raw_internal is None
                else InternalSubrouteEmbeddingSummary(**dict(raw_internal)),
                coverage=CoverageSummary(**raw_coverage),
            )
        )
    return RouteEmbeddingAudit(
        release_name=str(payload["release_name"]),
        match_level=str(payload["match_level"]),
        allow_leaf_extension=bool(payload["allow_leaf_extension"]),
        partial_min_reactions=payload["partial_min_reactions"],
        container_routes=int(payload["container_routes"]),
        container_route_signatures=int(payload["container_route_signatures"]),
        container_reaction_signatures=int(payload["container_reaction_signatures"]),
        query_sets=tuple(query_sets),
        ledger_rows=tuple(RouteEmbeddingLedgerRow.model_validate(row) for row in payload["ledger_rows"]),
    )
