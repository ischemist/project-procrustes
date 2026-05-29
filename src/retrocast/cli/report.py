from __future__ import annotations

import re

from rich.markup import escape
from rich.table import Table

from retrocast.models.analysis import AnalysisReport, MetricSummary

_SOLV_RATE = re.compile(r"^solv_(\d+)\[(.+)]_rate$")
_MRR_SOLV = re.compile(r"^mrr_solv_(\d+)\[(.+)]$")
_TOP_K = re.compile(r"^acceptable_reconstruction_top_(\d+)\[(.+)]$")


def create_analysis_table(report: AnalysisReport, *, title: str = "Analysis Results") -> Table:
    """Create the compact CLI table; stratified diagnostics stay in report.md."""
    table = Table(title=escape(title), header_style="bold magenta", show_lines=False)
    table.add_column("Metric", style="bold", min_width=32)
    table.add_column("Value", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("N", justify="right")

    solv_metrics = _matching_metrics(report.metrics, _SOLV_RATE)
    if solv_metrics:
        table.add_row("[dim]Solv-N hierarchy[/]", "", "", "")
        for name, metric, match in solv_metrics:
            table.add_row(_solv_label(match), _format_value(name, metric), _format_ci(name, metric), str(metric.count))

    mrr_metrics = _matching_metrics(report.metrics, _MRR_SOLV)
    if mrr_metrics:
        table.add_section()
        table.add_row("[dim]Rank within Solv-N hierarchy[/]", "", "", "")
        for name, metric, match in mrr_metrics:
            table.add_row(_mrr_label(match), _format_value(name, metric), _format_ci(name, metric), str(metric.count))

    top_k_metrics = _matching_metrics(report.metrics, _TOP_K)
    if top_k_metrics:
        table.add_section()
        table.add_row("[dim]Benchmark route reconstruction[/]", "", "", "")
        for name, metric, match in top_k_metrics:
            table.add_row(_top_k_label(match), _format_value(name, metric), _format_ci(name, metric), str(metric.count))

    return table


def generate_markdown_report(report: AnalysisReport, *, title: str = "Evaluation Report") -> str:
    lines = [f"# {title}", "", "## Overall", ""]
    lines.extend(_markdown_metric_table(report.metrics))
    if report.by_stratum:
        lines.extend(["", "## By Stratum", ""])
        for stratum in sorted(report.by_stratum):
            lines.extend([f"### {stratum}", ""])
            lines.extend(_markdown_metric_table(report.by_stratum[stratum]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_metric_table(metrics: dict[str, MetricSummary]) -> list[str]:
    lines = ["| Metric | Value | 95% CI | N |", "| --- | ---: | :---: | ---: |"]
    for name, metric, match in _ordered_metrics(metrics):
        lines.append(
            f"| {_display_metric_name(name, match)} | {_format_value(name, metric, rich=False)} | "
            f"{_format_ci(name, metric, rich=False)} | {metric.count} |"
        )
    return lines


def _ordered_metrics(metrics: dict[str, MetricSummary]) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    ordered = []
    for pattern in (_SOLV_RATE, _MRR_SOLV, _TOP_K):
        ordered.extend(_matching_metrics(metrics, pattern))
    return ordered


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
    if _SOLV_RATE.match(name):
        return _plain_solv_label(match)
    if _MRR_SOLV.match(name):
        return _plain_mrr_label(match)
    return _plain_top_k_label(match)


def _solv_label(match: re.Match[str]) -> str:
    return escape(_plain_solv_label(match))


def _mrr_label(match: re.Match[str]) -> str:
    return escape(_plain_mrr_label(match))


def _top_k_label(match: re.Match[str]) -> str:
    return escape(_plain_top_k_label(match))


def _plain_solv_label(match: re.Match[str]) -> str:
    return f"Solv-{match.group(1)}[{match.group(2)}]"


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
