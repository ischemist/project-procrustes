"""Metric constraints and compatibility analysis helpers."""

from importlib import import_module
from typing import Any

from retrocast.metrics.constraints import (
    RequiredLeavesChecker,
    RouteDepthChecker,
    StockTerminationChecker,
    TaskConstraintChecker,
    check_task_constraints,
)

_LAZY_EXPORTS = {
    "PairwiseComparison": "retrocast.metrics.ranking",
    "RankResult": "retrocast.metrics.ranking",
    "StratifiedMetricSummary": "retrocast.metrics.bootstrap",
    "check_reliability": "retrocast.metrics.bootstrap",
    "compute_metric_with_ci": "retrocast.metrics.bootstrap",
    "compute_paired_difference": "retrocast.metrics.ranking",
    "compute_pairwise_tournament": "retrocast.metrics.ranking",
    "compute_probabilistic_ranking": "retrocast.metrics.ranking",
    "summarize_targets": "retrocast.metrics.analysis",
    "summarize_values": "retrocast.metrics.bootstrap",
}

__all__ = [
    "PairwiseComparison",
    "RankResult",
    "RequiredLeavesChecker",
    "RouteDepthChecker",
    "StockTerminationChecker",
    "StratifiedMetricSummary",
    "TaskConstraintChecker",
    "check_reliability",
    "check_task_constraints",
    "compute_metric_with_ci",
    "compute_paired_difference",
    "compute_pairwise_tournament",
    "compute_probabilistic_ranking",
    "summarize_targets",
    "summarize_values",
]


def __getattr__(name: str) -> Any:
    """Load legacy analysis helpers only when a caller explicitly requests one."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
