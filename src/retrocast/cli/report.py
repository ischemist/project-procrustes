from __future__ import annotations

import re

from rich.markup import escape
from rich.table import Table

from retrocast.markdown import MarkdownRow, markdown_table
from retrocast.models.analysis import AnalysisReport, MetricSummary

_SOLV_RATE = re.compile(r"^solv_(\d+)\[(.+)]_rate$")
_MRR_SOLV = re.compile(r"^mrr_solv_(\d+)\[(.+)]$")
_TIER_VALIDITY = re.compile(r"^tier_(\d+)_validity_rate$")
_MRR_TIER = re.compile(r"^mrr_tier_(\d+)$")
_TOP_K = re.compile(r"^acceptable_reconstruction_top_(\d+)\[(.+)]$")


def create_analysis_table(report: AnalysisReport, *, title: str = "Analysis Results") -> Table:
    """Create the compact CLI table; stratified diagnostics stay in report.md."""
    table = Table(title=escape(title), header_style="bold magenta", show_lines=False)
    table.add_column("Metric", style="bold", min_width=32)
    table.add_column("Value", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("N", justify="right")
    table.add_column("Reliability", justify="center")

    hierarchy_metrics = _hierarchy_metrics(report.metrics)
    if hierarchy_metrics:
        table.add_row("[dim]Solv-N evaluation[/]", "", "", "", "")
        for name, metric, match in hierarchy_metrics:
            table.add_row(
                escape(_display_metric_name(name, match)),
                _format_value(name, metric),
                _format_ci(name, metric),
                str(metric.count),
                _format_reliability(metric),
            )

    top_k_metrics = _matching_metrics(report.metrics, _TOP_K)
    if top_k_metrics:
        table.add_section()
        table.add_row("[dim]Benchmark route reconstruction[/]", "", "", "", "")
        for name, metric, match in top_k_metrics:
            table.add_row(
                _top_k_label(match),
                _format_value(name, metric),
                _format_ci(name, metric),
                str(metric.count),
                _format_reliability(metric),
            )

    runtime_rows = _runtime_rows(report)
    if runtime_rows:
        table.add_section()
        table.add_row("[dim]Runtime[/]", "", "", "", "")
        for label, value in runtime_rows:
            table.add_row(label, value, "", str(report.runtime.timed_target_count), "")

    return table


def generate_markdown_report(report: AnalysisReport, *, title: str = "Evaluation Report") -> str:
    lines = [f"# {title}", "", "## Overall", ""]
    lines.extend(_markdown_metric_table(report.metrics))
    runtime_rows = _runtime_rows(report, rich=False)
    if runtime_rows:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                markdown_table(
                    ["Metric", "Seconds", "Timed Targets"],
                    [(label, value, report.runtime.timed_target_count) for label, value in runtime_rows],
                    align=["left", "right", "right"],
                ),
            ]
        )
    if report.by_stratum:
        lines.extend(["", "## By Stratum", ""])
        for stratum in sorted(report.by_stratum):
            lines.extend([f"### {stratum}", ""])
            lines.extend(_markdown_metric_table(report.by_stratum[stratum]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_metric_table(metrics: dict[str, MetricSummary]) -> list[str]:
    rows: list[MarkdownRow] = []
    for name, metric, match in _ordered_metrics(metrics):
        rows.append(
            (
                _display_metric_name(name, match),
                _format_value(name, metric, rich=False),
                _format_ci(name, metric, rich=False),
                metric.count,
                _format_reliability(metric, rich=False),
            )
        )
    return markdown_table(
        ["Metric", "Value", "95% CI", "N", "Flags"],
        rows,
        align=["left", "right", "center", "right", "center"],
    ).splitlines()


def _runtime_rows(report: AnalysisReport, *, rich: bool = True) -> list[tuple[str, str]]:
    runtime = report.runtime
    rows = []
    if runtime.total_wall_time is not None:
        rows.append(("Total wall time", _format_seconds(runtime.total_wall_time, rich=rich)))
    if runtime.mean_wall_time is not None:
        rows.append(("Mean wall time", _format_seconds(runtime.mean_wall_time, rich=rich)))
    if runtime.total_cpu_time is not None:
        rows.append(("Total CPU time", _format_seconds(runtime.total_cpu_time, rich=rich)))
    if runtime.mean_cpu_time is not None:
        rows.append(("Mean CPU time", _format_seconds(runtime.mean_cpu_time, rich=rich)))
    return rows


def _format_seconds(value: float, *, rich: bool) -> str:
    formatted = f"{value:.2f}s"
    return f"[green]{formatted}[/]" if rich else formatted


def _format_reliability(metric: MetricSummary, *, rich: bool = True) -> str:
    reliability = metric.reliability
    if reliability is None:
        return ""
    if reliability.code == "OK":
        return "[green]OK[/]" if rich else "OK"
    code = escape(reliability.code) if rich else reliability.code
    return f"[yellow]{code}[/]" if rich else code


def _ordered_metrics(metrics: dict[str, MetricSummary]) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    ordered = _hierarchy_metrics(metrics)
    for pattern in (_TOP_K,):
        ordered.extend(_matching_metrics(metrics, pattern))
    return ordered


def _hierarchy_metrics(metrics: dict[str, MetricSummary]) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    matches: list[tuple[str, MetricSummary, re.Match[str]]] = []
    for pattern in (_TIER_VALIDITY, _SOLV_RATE, _MRR_TIER, _MRR_SOLV):
        matches.extend(_matching_metrics(metrics, pattern))
    return sorted(matches, key=_hierarchy_sort_key)


def _hierarchy_sort_key(item: tuple[str, MetricSummary, re.Match[str]]) -> tuple[int, int, str]:
    name, _, match = item
    if _TIER_VALIDITY.match(name):
        return (int(match.group(1)), 0, "")
    if _SOLV_RATE.match(name):
        return (int(match.group(1)), 1, match.group(2))
    if _MRR_TIER.match(name):
        return (int(match.group(1)), 2, "")
    return (int(match.group(1)), 3, match.group(2))


def _matching_metrics(
    metrics: dict[str, MetricSummary], pattern: re.Pattern[str]
) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    matches = []
    for name, metric in metrics.items():
        match = pattern.match(name)
        if match is not None:
            matches.append((name, metric, match))
    return sorted(matches, key=lambda item: int(item[2].group(1)))


def _display_metric_name(name: str, match: re.Match[str]) -> str:
    if _TIER_VALIDITY.match(name):
        return _plain_tier_validity_label(match)
    if _SOLV_RATE.match(name):
        return _plain_solv_label(match)
    if _MRR_TIER.match(name):
        return _plain_mrr_tier_label(match)
    if _MRR_SOLV.match(name):
        return _plain_mrr_label(match)
    return _plain_top_k_label(match)


def _top_k_label(match: re.Match[str]) -> str:
    return escape(_plain_top_k_label(match))


def _plain_solv_label(match: re.Match[str]) -> str:
    return f"Solv-{match.group(1)}[{match.group(2)}]"


def _plain_tier_validity_label(match: re.Match[str]) -> str:
    return f"Tier-{match.group(1)} Validity"


def _plain_mrr_tier_label(match: re.Match[str]) -> str:
    return f"MRR Tier-{match.group(1)}"


def _plain_mrr_label(match: re.Match[str]) -> str:
    return f"MRR Solv-{match.group(1)}[{match.group(2)}]"


def _plain_top_k_label(match: re.Match[str]) -> str:
    return f"Top-{match.group(1)}"


def _format_value(name: str, metric: MetricSummary, *, rich: bool = True) -> str:
    value = f"{metric.value:.3f}" if name.startswith("mrr_") else f"{metric.value:.1%}"
    return f"[green]{value}[/]" if rich else value


def _format_ci(name: str, metric: MetricSummary, *, rich: bool = True) -> str:
    if metric.ci_low is None or metric.ci_high is None:
        return ""
    if name.startswith("mrr_"):
        value = f"[{metric.ci_low:.3f}, {metric.ci_high:.3f}]"
    else:
        value = f"[{metric.ci_low:.1%}, {metric.ci_high:.1%}]"
    return f"[green]{value}[/]" if rich else value
