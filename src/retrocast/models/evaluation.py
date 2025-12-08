from typing import Any

from pydantic import BaseModel, Field


class ScoredRoute(BaseModel):
    """
    A single predicted route, stripped of chemistry, enriched with metrics.
    This is the 'atom' of statistical analysis.
    """

    rank: int
    is_solved: bool
    matches_acceptable: bool  # Does this route match any acceptable route?
    matched_acceptable_index: int | None = None  # Index of matched acceptable route (if any)
    # We can add more here later: 'num_steps', 'confidence_score', etc.


class TargetEvaluation(BaseModel):
    """
    The result of evaluating one target against one stock.
    """

    target_id: str

    # We store ALL scored routes, sorted by rank.
    # This allows O(1) slicing for top-k during stats.
    routes: list[ScoredRoute] = Field(default_factory=list)

    # Shortcuts for the lazy
    is_solvable: bool = False  # At least one route is solved
    acceptable_rank: int | None = None  # Rank of first solved acceptable match (None if not found)

    # Properties used for stratification (route length, convergence)
    stratification_length: int | None = None
    stratification_is_convergent: bool | None = None

    # Runtime metrics for this target (in seconds)
    wall_time: float | None = None
    cpu_time: float | None = None


class EvaluationResults(BaseModel):
    """
    The complete dump of a scoring run.
    """

    model_name: str
    benchmark_name: str
    stock_name: str
    has_acceptable_routes: bool  # Whether the benchmark has acceptable routes (not whether model found them)

    # Map target_id -> Evaluation
    results: dict[str, TargetEvaluation] = Field(default_factory=dict)

    # Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)
