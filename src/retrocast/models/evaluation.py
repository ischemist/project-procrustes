from typing import Any

from pydantic import BaseModel, Field


class ScoredRoute(BaseModel):
    """
    A single predicted route, stripped of chemistry, enriched with metrics.
    This is the 'atom' of statistical analysis.
    """

    rank: int
    is_solved: bool
    is_gt_match: bool  # Does this route match the ground truth?
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
    gt_rank: int | None = None  # Rank of first solved GT match (None if not found)

    # Metadata copied from BenchmarkTarget for easy stratification
    route_length: int | None
    is_convergent: bool | None


class EvaluationResults(BaseModel):
    """
    The complete dump of a scoring run.
    """

    model_name: str
    benchmark_name: str
    stock_name: str
    has_ground_truth: bool  # Whether the benchmark has ground truth routes (not whether model found them)

    # Map target_id -> Evaluation
    results: dict[str, TargetEvaluation] = Field(default_factory=dict)

    # Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)
