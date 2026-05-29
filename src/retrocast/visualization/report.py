from __future__ import annotations

from typing import Any

from rich.table import Table

from retrocast.cli.report import create_analysis_table, generate_markdown_report


def create_ranking_table(ranking_results: list[Any], metric_label: str) -> Table:
    table = Table(title=f"Probabilistic Ranking based on {metric_label}", header_style="bold magenta", expand=True)
    table.add_column("Model", style="bold")
    table.add_column("Expected Rank", justify="right")
    table.add_column("Prob. #1", justify="right")
    table.add_column("Prob. Top-3", justify="right")
    for result in ranking_results:
        prob_first = result.rank_probs.get(1, 0.0)
        prob_top3 = sum(result.rank_probs.get(rank, 0.0) for rank in (1, 2, 3))
        style = "green" if prob_first > 0.5 else ""
        table.add_row(
            result.model_name,
            f"{result.expected_rank:.2f}",
            f"{prob_first:.1%}",
            f"{prob_top3:.1%}",
            style=style,
        )
    return table


def create_tournament_table(comparisons: list[Any], model_names: list[str]) -> Table:
    table = Table(title="Tournament Results (row - column)", show_lines=True, header_style="bold")
    table.add_column("Model", style="bold cyan")
    for model_name in model_names:
        table.add_column(model_name, justify="center")
    comparison_map = {(comparison.model_a, comparison.model_b): comparison for comparison in comparisons}
    for row_model in model_names:
        cells = [row_model]
        for column_model in model_names:
            if row_model == column_model:
                cells.append("[dim]-[/]")
                continue
            comparison = comparison_map.get((row_model, column_model))
            if comparison is None:
                cells.append("?")
                continue
            value = -comparison.diff_mean
            if comparison.is_significant:
                color = "green" if value > 0 else "red"
                cells.append(f"[bold {color}]{value:+.1%}[/]")
            else:
                cells.append(f"[dim]{value:+.1%}[/]")
        table.add_row(*cells)
    return table


def create_stability_table(metrics_summary: dict[str, dict[str, float]], seed_deviations: list[tuple]) -> tuple[Table, Table]:
    stats_table = Table(title="Stability Statistics", header_style="bold cyan")
    stats_table.add_column("Metric")
    stats_table.add_column("Mean (%)", justify="right")
    stats_table.add_column("Std Dev", justify="right")
    for metric_name, stats in metrics_summary.items():
        stats_table.add_row(metric_name, f"{stats['mean']:.2f}", f"{stats['std']:.3f}")

    ranking_table = Table(title="Seed Representativeness (lowest deviation is best)", header_style="bold magenta")
    ranking_table.add_column("Rank", justify="right")
    ranking_table.add_column("Seed", justify="center")
    ranking_table.add_column("Deviation Score", justify="right")
    ranking_table.add_column("Z-Scores (Top1, Solv, Top10)", justify="right")
    for index, values in enumerate(seed_deviations[:5], 1):
        seed, deviation, z_top1, z_solv, z_top10 = values
        ranking_table.add_row(str(index), str(seed), f"{deviation:.4f}", f"({z_top1:+.1f}, {z_solv:+.1f}, {z_top10:+.1f})")
    return stats_table, ranking_table


__all__ = [
    "create_analysis_table",
    "create_ranking_table",
    "create_stability_table",
    "create_tournament_table",
    "generate_markdown_report",
]
