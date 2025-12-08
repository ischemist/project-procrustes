"""
Reporting and text-based visualization utilities.

This module handles the formatting of results into text tables (Markdown/Rich).
"""

from rich.table import Table

from retrocast.models.stats import ModelComparison, ModelStatistics, StratifiedMetric


def create_paired_comparison_table(
    baseline_name: str, benchmark_name: str, comparisons: list[ModelComparison]
) -> Table:
    """
    Creates a Rich table summarizing paired comparisons.
    Applies conditional styling based on statistical significance.
    """
    table = Table(
        title=f"Paired Comparison vs Baseline: [bold]{baseline_name}[/]\nBenchmark: {benchmark_name}",
        header_style="bold cyan",
        expand=True,
        show_lines=False,
    )

    table.add_column("Challenger", style="bold")
    table.add_column("Metric")
    table.add_column("Diff (Chal - Base)", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("Sig?", justify="center")

    current_challenger = None

    for comp in comparisons:
        # Add section break between different challengers
        if current_challenger is not None and comp.model_b != current_challenger:
            table.add_section()
        current_challenger = comp.model_b

        # Formatting Logic
        diff_str = f"{comp.diff_mean:+.1%}"
        ci_str = f"[{comp.diff_ci_lower:+.1%}, {comp.diff_ci_upper:+.1%}]"

        style = ""
        sig_icon = ""

        if comp.is_significant:
            sig_icon = "✅"
            # Positive Diff = Challenger (B) > Baseline (A)
            if comp.diff_mean > 0:
                style = "green"
            else:
                style = "red"
        else:
            # Not significant - dim it
            style = "dim"
            sig_icon = "-"

        table.add_row(comp.model_b, comp.metric, diff_str, ci_str, sig_icon, style=style)

    return table


def create_ranking_table(ranking_results: list, metric_label: str) -> Table:
    """Creates a pretty table for ranking results."""
    table = Table(title=f"Probabilistic Ranking based on {metric_label}", header_style="bold magenta", expand=True)
    table.add_column("Model", style="bold")
    table.add_column("Expected Rank", justify="right")
    table.add_column("Prob. of being #1", justify="right")
    table.add_column("Prob. of being Top-3", justify="right")

    for r in ranking_results:
        prob_first = r.rank_probs.get(1, 0.0)

        # Calculate prob of being in top 3
        prob_top3 = sum(r.rank_probs.get(i, 0.0) for i in [1, 2, 3])

        # Highlight the winner
        style = "green" if prob_first > 0.5 else ""

        table.add_row(r.model_name, f"{r.expected_rank:.2f}", f"{prob_first:.1%}", f"{prob_top3:.1%}", style=style)
    return table


def create_tournament_table(comparisons: list[ModelComparison], model_names: list[str]) -> Table:
    """Creates the tournament matrix table (from script 05)."""
    table = Table(title="Tournament Results (Row - Col)", box=None, show_lines=True, header_style="bold")
    table.add_column("Model", style="bold cyan")
    for m in model_names:
        table.add_column(m, justify="center")

    comp_map = {(c.model_a, c.model_b): c for c in comparisons}

    for row_model in model_names:
        row_cells = [row_model]
        for col_model in model_names:
            if row_model == col_model:
                row_cells.append("[dim]-[/]")
                continue

            comp = comp_map.get((row_model, col_model))
            if not comp:
                row_cells.append("?")
                continue

            val = comp.diff_mean
            if not comp.is_significant:
                txt = f"[dim]{val:+.1%}[/]"
            else:
                color = "green" if val > 0 else "red"
                txt = f"[bold {color}]{val:+.1%}[/]"
            row_cells.append(txt)

        table.add_row(*row_cells)

    return table


def create_stability_table(metrics_summary: dict, seed_deviations: list) -> tuple[Table, Table]:
    """Creates stability analysis tables (from script 06). Returns (stats_table, ranking_table)."""
    # Table 1: Stats
    t1 = Table(title="Stability Statistics", header_style="bold cyan")
    t1.add_column("Metric")
    t1.add_column("Mean (%)", justify="right")
    t1.add_column("Std Dev", justify="right")
    for m, stats in metrics_summary.items():
        t1.add_row(m, f"{stats['mean']:.2f}", f"{stats['std']:.3f}")

    # Table 2: Seeds
    t2 = Table(title="Seed Representativeness (Lowest Deviation is Best)", header_style="bold magenta")
    t2.add_column("Rank", justify="right")
    t2.add_column("Seed", justify="center")
    t2.add_column("Deviation Score", justify="right")
    t2.add_column("Z-Scores (Top1, Solv, Top10)", justify="right")

    for i, (seed, dev, z1, zs, z10) in enumerate(seed_deviations[:5], 1):
        t2.add_row(str(i), str(seed), f"{dev:.4f}", f"({z1:+.1f}, {zs:+.1f}, {z10:+.1f})")

    return t1, t2


def create_single_model_summary_table(stats: ModelStatistics, visible_k: list[int] | None = None) -> Table:
    """
    Creates a high-level summary table for a single model analysis.
    Used in the CLI analysis workflow.

    Args:
        stats: The model statistics object.
        visible_k: Optional list of K values to include.
                   If None, defaults to [1, 5, 10, 50].
    """
    table = Table(
        title=f"Analysis Results: [bold cyan]{stats.model_name}[/]", header_style="bold magenta", show_lines=False
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("N", justify="right")
    table.add_column("Reliability", justify="center")

    # Helper to add rows
    def _add(name: str, res):
        color = "green" if res.reliability.code == "OK" else "yellow"
        rel_icon = "✅" if res.reliability.code == "OK" else f"⚠️ {res.reliability.code}"

        table.add_row(
            name,
            f"[{color}]{res.value:.1%}[/]",
            f"[{color}][{res.ci_lower:.1%}, {res.ci_upper:.1%}][/]",
            str(res.n_samples),
            rel_icon,
        )

    if visible_k is None:
        visible_k = [1, 5, 10, 50]

    _add("Solvability", stats.solvability.overall)

    # Add Top-K
    for k in sorted(stats.top_k_accuracy.keys()):
        if k in visible_k:
            _add(f"Top-{k}", stats.top_k_accuracy[k].overall)

    # Add runtime metrics if available
    if stats.total_wall_time is not None or stats.total_cpu_time is not None:
        table.add_section()

        if stats.total_wall_time is not None:
            table.add_row(
                "Total Wall Time",
                f"[cyan]{stats.total_wall_time:.2f}s[/]",
                "",
                "",
                "",
            )
            if stats.mean_wall_time is not None:
                table.add_row(
                    "Mean Wall Time",
                    f"[cyan]{stats.mean_wall_time:.2f}s[/]",
                    "",
                    "",
                    "",
                )

        if stats.total_cpu_time is not None:
            table.add_row(
                "Total CPU Time",
                f"[cyan]{stats.total_cpu_time:.2f}s[/]",
                "",
                "",
                "",
            )
            if stats.mean_cpu_time is not None:
                table.add_row(
                    "Mean CPU Time",
                    f"[cyan]{stats.mean_cpu_time:.2f}s[/]",
                    "",
                    "",
                    "",
                )

    return table


def format_metric_table(stats: StratifiedMetric) -> str:
    """Markdown table generator with reliability flags."""
    lines = []

    # Add warning for Overall if needed
    flag_icon = ""
    if stats.overall.reliability.code != "OK":
        flag_icon = f" ⚠️ {stats.overall.reliability.code}"

    lines.append(f"**Overall**: {stats.overall.value:.1%} (N={stats.overall.n_samples}){flag_icon}")
    lines.append(f"CI: [{stats.overall.ci_lower:.1%}, {stats.overall.ci_upper:.1%}]")

    if stats.overall.reliability.code != "OK":
        lines.append(f"*{stats.overall.reliability.message}*")

    lines.append("")

    if not stats.by_group:
        return "\n".join(lines)

    # Add "Reliability" column
    lines.append("| Group | N | Value | 95% CI | Flags |")
    lines.append("|-------|---|-------|--------|-------|")

    sorted_keys = sorted(stats.by_group.keys())

    for key in sorted_keys:
        res = stats.by_group[key]
        ci = f"[{res.ci_lower:.1%}, {res.ci_upper:.1%}]"
        val = f"{res.value:.1%}"

        # Determine flag
        flag = ""
        if res.reliability.code == "LOW_N":
            flag = "⚠️ Low N"
        elif res.reliability.code == "EXTREME_P":
            flag = "⚠️ Boundary"

        lines.append(f"| {key} | {res.n_samples} | {val} | {ci} | {flag} |")

    return "\n".join(lines)


def generate_markdown_report(stats: ModelStatistics, visible_k: list[int] | None = None) -> str:
    """
    Generates a full markdown report.

    Args:
        stats: The model statistics object.
        visible_k: Optional list of K values to include.
                   If None, defaults to [1, 5, 10, 50].
    """
    if visible_k is None:
        visible_k = [1, 5, 10, 50]
    sections = [
        f"# Evaluation Report: {stats.model_name}",
        f"**Benchmark**: {stats.benchmark}",
        f"**Stock**: {stats.stock}",
        "",
    ]

    # Add runtime metrics if available
    if stats.total_wall_time is not None or stats.total_cpu_time is not None:
        sections.append("## Runtime Metrics")
        sections.append("")
        if stats.total_wall_time is not None:
            sections.append(f"- **Total Wall Time**: {stats.total_wall_time:.2f}s")
            if stats.mean_wall_time is not None:
                sections.append(f"- **Mean Wall Time**: {stats.mean_wall_time:.2f}s per target")
        if stats.total_cpu_time is not None:
            sections.append(f"- **Total CPU Time**: {stats.total_cpu_time:.2f}s")
            if stats.mean_cpu_time is not None:
                sections.append(f"- **Mean CPU Time**: {stats.mean_cpu_time:.2f}s per target")
        sections.append("")

    sections.extend(
        [
            "## Solvability",
            format_metric_table(stats.solvability),
            "",
        ]
    )

    available_k = sorted(stats.top_k_accuracy.keys())

    for k in available_k:
        if k in visible_k:
            sections.append(f"## Top-{k} Accuracy")
            sections.append(format_metric_table(stats.top_k_accuracy[k]))
            sections.append("")

    return "\n".join(sections)
