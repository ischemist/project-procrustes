from __future__ import annotations

from pydantic import BaseModel, Field

from retrocast.models.chem import Route


class BenchmarkCollectionStats(BaseModel):
    """Structured summary of benchmark collection outcomes."""

    total_routes: int = 0
    matched_by_target_hint: int = 0
    matched_by_canonical_smiles: int = 0
    unmatched_routes: int = 0
    ambiguous_routes: int = 0
    duplicate_routes_dropped: int = 0
    final_unique_routes_saved: int = 0


class CollectedBenchmarkRoutes(BaseModel):
    """Benchmark-keyed routes plus collection statistics."""

    routes_by_target: dict[str, list[Route]] = Field(default_factory=dict)
    stats: BenchmarkCollectionStats = Field(default_factory=BenchmarkCollectionStats)
