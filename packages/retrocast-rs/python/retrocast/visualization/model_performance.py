from __future__ import annotations

from typing import Any

from retrocast.visualization.plots import (
    plot_comparison,
    plot_diagnostics,
    plot_overall_summary,
    plot_pairwise_matrix,
    plot_ranking,
)


def plot_single_model_diagnostics(stats: Any):
    return plot_diagnostics(stats)


def plot_multi_model_comparison(models_stats: list[Any], metric_type: str = "Top-1", k: int = 1):
    if metric_type == "Solvability":
        return plot_comparison(models_stats, metric_type="Solvability", k=k)
    if metric_type.startswith("Top-"):
        k = int(metric_type.split("-", 1)[1])
    return plot_comparison(models_stats, metric_type="Top-K", k=k)


def plot_overall_comparison(models_stats: list[Any]):
    return plot_overall_summary(models_stats, top_k_values=[1, 5, 10])


def plot_probabilistic_ranking(ranking: list[Any], metric_name: str):
    return plot_ranking(ranking, metric_name)


__all__ = [
    "plot_multi_model_comparison",
    "plot_overall_comparison",
    "plot_pairwise_matrix",
    "plot_probabilistic_ranking",
    "plot_single_model_diagnostics",
]
