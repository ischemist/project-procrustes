from retrocast.models.stats import ModelStatistics, StratifiedMetric


def format_metric_table(stats: StratifiedMetric) -> str:
    """Markdown table generator."""
    lines = []
    lines.append(f"**Overall**: {stats.overall.value:.1%} (N={stats.overall.n_samples})")
    lines.append(f"CI: [{stats.overall.ci_lower:.1%}, {stats.overall.ci_upper:.1%}]")
    lines.append("")

    if not stats.by_group:
        return "\n".join(lines)

    lines.append("| Group | N | Value | 95% CI |")
    lines.append("|-------|---|-------|--------|")

    # Sort by group key (assuming int depth)
    sorted_keys = sorted(stats.by_group.keys())

    for key in sorted_keys:
        res = stats.by_group[key]
        ci = f"[{res.ci_lower:.1%}, {res.ci_upper:.1%}]"
        val = f"{res.value:.1%}"
        lines.append(f"| {key} | {res.n_samples} | {val} | {ci} |")

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
