"""Schema v2 workflow APIs."""

from retrocast.models.evaluation import AcceptableRouteMatch
from retrocast.workflow.adapt import adapt_candidates, adapt_route, adapt_routes
from retrocast.workflow.analyze import analyze
from retrocast.workflow.collect import (
    CollectedCandidates,
    CollectedRoutes,
    collect_candidates,
    collect_routes,
)
from retrocast.workflow.ingest import ingest_candidates, ingest_routes
from retrocast.workflow.score import score

__all__ = [
    "AcceptableRouteMatch",
    "CollectedCandidates",
    "CollectedRoutes",
    "adapt_candidates",
    "adapt_route",
    "adapt_routes",
    "analyze",
    "collect_candidates",
    "collect_routes",
    "ingest_candidates",
    "ingest_routes",
    "score",
]
