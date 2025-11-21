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


def generate_markdown_report(stats: ModelStatistics) -> str:
    """Full report."""
    sections = [
        f"# Evaluation Report: {stats.model_name}",
        f"**Benchmark**: {stats.benchmark}",
        f"**Stock**: {stats.stock}",
        "",
        "## Solvability",
        format_metric_table(stats.solvability),
        "",
    ]

    for k in sorted(stats.top_k_accuracy.keys()):
        sections.append(f"## Top-{k} Accuracy")
        sections.append(format_metric_table(stats.top_k_accuracy[k]))
        sections.append("")

    return "\n".join(sections)
