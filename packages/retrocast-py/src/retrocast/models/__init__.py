"""Schema v2 models."""

from retrocast.models.analysis import AnalysisReport, MetricSummary, ReliabilityFlag
from retrocast.models.candidates import Candidate, FailureRecord
from retrocast.models.evaluation import (
    AcceptableRouteMatch,
    CheckResult,
    CheckStatus,
    ConstraintResult,
    Evaluation,
    ReactionValidity,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.models.route import (
    InChIKeyLevel,
    Molecule,
    MoleculeId,
    Reaction,
    ReactionId,
    Route,
    RoutePath,
)
from retrocast.models.task import (
    Benchmark,
    RequiredLeavesConstraint,
    RouteDepthConstraint,
    StockTerminationConstraint,
    Target,
    Task,
    TaskConstraint,
)

__all__ = [
    "AnalysisReport",
    "AcceptableRouteMatch",
    "Benchmark",
    "Candidate",
    "CheckResult",
    "CheckStatus",
    "ConstraintResult",
    "Evaluation",
    "FailureRecord",
    "InChIKeyLevel",
    "Molecule",
    "MoleculeId",
    "MetricSummary",
    "Reaction",
    "ReactionId",
    "ReactionValidity",
    "ReliabilityFlag",
    "Route",
    "RoutePath",
    "RouteValidity",
    "RouteDepthConstraint",
    "ScoredCandidate",
    "StockTerminationConstraint",
    "Target",
    "TargetResult",
    "Task",
    "TaskConstraint",
    "Tier",
    "TierResult",
    "RequiredLeavesConstraint",
]
