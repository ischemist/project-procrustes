from __future__ import annotations

import re

from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from retrocast.markdown import MarkdownAlign, MarkdownRow, markdown_table
from retrocast.models.analysis import AnalysisReport, MetricSummary

_SOLV_RATE = re.compile(r"^solv_(\d+)\[(.+)]_rate$")
_MRR_SOLV = re.compile(r"^mrr_solv_(\d+)\[(.+)]$")
_TIER_VALIDITY = re.compile(r"^tier_(\d+)_validity_rate$")
_MRR_TIER = re.compile(r"^mrr_tier_(\d+)$")
_TOP_K = re.compile(r"^acceptable_reconstruction_top_(\d+)\[(.+)]$")
_ROOT_TOP_K = re.compile(r"^acceptable_root_reconstruction_top_(\d+)\[(.+)]$")
_GIVEN_ROOT_TOP_K = re.compile(r"^acceptable_reconstruction_given_root_top_(\d+)\[(.+)]$")
_PREFIX_TOP_K = re.compile(r"^acceptable_prefix_reconstruction_depth_(\d+)_top_(\d+)\[(.+)]$")
_DISTINCT_ROOT_TOP_K = re.compile(r"^distinct_root_reactions_top_(\d+)\[(.+)]$")


def create_analysis_table(
    report: AnalysisReport,
    *,
    title: str = "analysis results",
    subtitle: str | None = None,
) -> Group:
    renderables: list[RenderableType] = [Text.from_markup(f"[bold magenta]{escape(title)}[/]")]
    if subtitle is not None:
        renderables.append(Text.from_markup(f"[dim]{escape(subtitle)}[/]"))
    note = _terminal_report_note(report)
    if note is not None:
        renderables.append(Text.from_markup(note))

    solv_table = _create_solv_table(report)
    if solv_table is not None:
        renderables.append(solv_table)

    reconstruction_table = _create_reconstruction_table(report.metrics)
    if reconstruction_table is not None:
        renderables.append(reconstruction_table)

    runtime_table = _create_runtime_table(report)
    if runtime_table is not None:
        renderables.append(runtime_table)

    return Group(*renderables)


def _create_solv_table(
    report: AnalysisReport,
) -> Table | None:
    metrics = report.metrics
    tiers = _solv_tiers(metrics)
    if not tiers:
        return None

    table = Table(
        title="Solv-N Evaluation",
        title_style="bold magenta",
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("Tier", style="bold", justify="right", no_wrap=True)
    table.add_column("Task", no_wrap=True)
    table.add_column("Valid", justify="center", no_wrap=True)
    table.add_column("Solv", justify="center", no_wrap=True)
    table.add_column("MRR Valid", justify="center", no_wrap=True)
    table.add_column("MRR Solv", justify="center", no_wrap=True)

    for tier in tiers:
        labels = _solv_labels(metrics, tier)
        if not labels:
            labels = [""]
        for label in labels:
            table.add_row(
                str(tier),
                label,
                _metric_cell(
                    f"tier_{tier}_validity_rate",
                    metrics.get(f"tier_{tier}_validity_rate"),
                ),
                _metric_cell(
                    f"solv_{tier}[{label}]_rate",
                    metrics.get(f"solv_{tier}[{label}]_rate"),
                )
                if label
                else "",
                _metric_cell(
                    f"mrr_tier_{tier}",
                    metrics.get(f"mrr_tier_{tier}"),
                ),
                _metric_cell(
                    f"mrr_solv_{tier}[{label}]",
                    metrics.get(f"mrr_solv_{tier}[{label}]"),
                )
                if label
                else "",
            )
    return table


def _create_reconstruction_table(
    metrics: dict[str, MetricSummary],
) -> Table | None:
    ks = _diagnostic_ks(metrics)
    if not ks:
        return None

    depths = _diagnostic_prefix_depths(metrics)

    def metric_cell(pattern: re.Pattern[str], k: int) -> str:
        for name, metric, match in _matching_metrics(metrics, pattern):
            if int(match.group(1)) == k:
                return _metric_cell(name, metric, compact_ci=True)
        return ""

    def prefix_cell(k: int, depth: int) -> str:
        for name, metric, match in _matching_metrics(metrics, _PREFIX_TOP_K):
            if int(match.group(1)) == depth and int(match.group(2)) == k:
                return _metric_cell(name, metric, compact_ci=True)
        return ""

    table = Table(
        title="Benchmark Route Reconstruction",
        title_style="bold magenta",
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("Metric", style="bold", min_width=12, no_wrap=True)
    for k in ks:
        table.add_column(f"Top-{k}", justify="center", no_wrap=True)

    rows = [
        ("Full route", [metric_cell(_TOP_K, k) for k in ks]),
        ("Root reaction", [metric_cell(_ROOT_TOP_K, k) for k in ks]),
        ("Route | root", [metric_cell(_GIVEN_ROOT_TOP_K, k) for k in ks]),
        *[(f"Prefix {depth}", [prefix_cell(k, depth) for k in ks]) for depth in depths],
        ("Unique roots", [metric_cell(_DISTINCT_ROOT_TOP_K, k) for k in ks]),
    ]
    for label, cells in rows:
        table.add_row(label, *cells)
    return table


def _create_runtime_table(report: AnalysisReport) -> Table | None:
    runtime_rows = _runtime_rows(report)
    if not runtime_rows:
        return None

    table = Table(
        title="Runtime",
        title_style="bold magenta",
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Wall", justify="right")
    table.add_column("CPU", justify="right")
    for label, wall_time, cpu_time in runtime_rows:
        table.add_row(label, wall_time, cpu_time)
    return table


def generate_markdown_report(report: AnalysisReport, *, title: str = "Evaluation Report") -> str:
    lines = [f"# {title}", "", "## Overall", ""]
    note = _solv_note(report)
    if note:
        lines.extend([note, ""])
    lines.extend(_markdown_headline_metric_table(report.metrics))
    lines.extend(_markdown_reconstruction_diagnostics(report.metrics))
    runtime_rows = _runtime_rows(report, rich=False)
    if runtime_rows:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                markdown_table(
                    ["Metric", "Wall", "CPU"],
                    runtime_rows,
                    align=["left", "right", "right"],
                ),
            ]
        )
    if report.by_stratum:
        lines.extend(["", "## By Stratum", ""])
        for stratum in sorted(report.by_stratum):
            lines.extend([f"### {stratum}", ""])
            lines.extend(_markdown_headline_metric_table(report.by_stratum[stratum]))
            lines.extend(_markdown_reconstruction_diagnostics(report.by_stratum[stratum], heading_level=4))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _solv_note(report: AnalysisReport) -> str:
    parts = []
    target_count = _target_count(report.metrics)
    if target_count is not None:
        parts.append(f"n={target_count} targets")

    if report.bootstrap_resamples is not None:
        parts.append(f"ci: 95% bootstrap, {report.bootstrap_resamples:,} resamples")
    else:
        parts.append("ci: 95% bootstrap")

    legend = _reliability_legend([metric for metric in report.metrics.values()])
    if legend:
        parts.append(legend)
    return " | ".join(parts)


def _terminal_report_note(report: AnalysisReport) -> str | None:
    lines = []
    lines.extend(f"[dim]{escape(line)}[/]" for line in _terminal_note_lines(report))
    return "\n".join(lines) if lines else None


def _terminal_note_lines(report: AnalysisReport) -> list[str]:
    parts = []
    target_count = _target_count(report.metrics)
    if target_count is not None:
        parts.append(f"n={target_count} targets")

    if report.bootstrap_resamples is not None:
        parts.append(f"ci=95% bootstrap, {report.bootstrap_resamples:,} resamples")
    else:
        parts.append("ci=95% bootstrap")

    flag_legend = _reliability_legend_parts([metric for metric in report.metrics.values()])
    if flag_legend:
        parts.append("flags: " + "; ".join(f"{symbol} {description}" for symbol, description in flag_legend))

    return [" | ".join(parts)]


def _plain_reconstruction_note(metrics: dict[str, MetricSummary]) -> str:
    notes = []
    legend = _reliability_legend(_reconstruction_metric_values(metrics))
    if legend:
        notes.append(legend)
    return " | ".join(notes)


def _markdown_headline_metric_table(metrics: dict[str, MetricSummary]) -> list[str]:
    rows: list[MarkdownRow] = []
    for name, metric, match in _headline_metrics(metrics):
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


def _markdown_reconstruction_diagnostics(
    metrics: dict[str, MetricSummary],
    *,
    heading_level: int = 2,
) -> list[str]:
    ks = _diagnostic_ks(metrics)
    if not ks:
        return []

    heading = "#" * heading_level
    lines = ["", f"{heading} Reconstruction Diagnostics", ""]
    note = _plain_reconstruction_note(metrics)
    if note:
        lines.extend([note, ""])

    def diagnostic_cell(pattern: re.Pattern[str], k: int, *, include_n: bool = False) -> str:
        for name, metric, match in _matching_metrics(metrics, pattern):
            if int(match.group(1)) != k:
                continue
            value = _format_value(name, metric, rich=False)
            ci = _format_ci(name, metric, rich=False)
            cell = f"{value} {ci}" if ci else value
            return f"{cell} (n={metric.count})" if include_n else cell
        return ""

    def prefix_cell(k: int, depth: int) -> str:
        for name, metric, match in _matching_metrics(metrics, _PREFIX_TOP_K):
            if int(match.group(1)) != depth or int(match.group(2)) != k:
                continue
            value = _format_value(name, metric, rich=False)
            ci = _format_ci(name, metric, rich=False)
            return f"{value} {ci}" if ci else value
        return ""

    top_k_rows = []
    for k in ks:
        top_k_rows.append(
            (
                f"Top-{k}",
                diagnostic_cell(_TOP_K, k),
                diagnostic_cell(_ROOT_TOP_K, k),
                diagnostic_cell(_GIVEN_ROOT_TOP_K, k, include_n=True),
                diagnostic_cell(_DISTINCT_ROOT_TOP_K, k),
            )
        )
    lines.extend(
        markdown_table(
            ["K", "Full route", "Root reaction", "Route given root", "Distinct roots"],
            top_k_rows,
            align=["left", "right", "right", "right", "right"],
        ).splitlines()
    )

    depths = _diagnostic_prefix_depths(metrics)
    if depths:
        prefix_rows = []
        for k in ks:
            prefix_rows.append(tuple([f"Top-{k}", *[prefix_cell(k, depth) for depth in depths]]))
        prefix_align: list[MarkdownAlign] = ["left"]
        for _ in depths:
            prefix_align.append("right")
        lines.extend(["", f"{heading} Prefix Reconstruction", ""])
        lines.extend(
            markdown_table(
                ["K", *[f"Depth {depth}" for depth in depths]],
                prefix_rows,
                align=prefix_align,
            ).splitlines()
        )

    return lines


def _runtime_rows(report: AnalysisReport, *, rich: bool = True) -> list[tuple[str, str, str]]:
    runtime = report.runtime
    rows = [
        (
            "Total time",
            _format_duration(runtime.total_wall_time),
            _format_duration(runtime.total_cpu_time),
        ),
        (
            "Per target",
            _format_duration(runtime.mean_wall_time),
            _format_duration(runtime.mean_cpu_time),
        ),
        (
            "Per 1M targets (projected)",
            _format_duration(_project_runtime(runtime.mean_wall_time)),
            _format_duration(_project_runtime(runtime.mean_cpu_time)),
        ),
    ]
    rows = [row for row in rows if row[1] or row[2]]
    if rich:
        return [_dim_runtime_projection(row) if "projected" in row[0] else row for row in rows]
    return rows


def _project_runtime(seconds_per_target: float | None) -> float | None:
    if seconds_per_target is None:
        return None
    return seconds_per_target * 1_000_000


def _dim_runtime_projection(row: tuple[str, str, str]) -> tuple[str, str, str]:
    label, wall_time, cpu_time = row
    return (f"[dim]{label}[/]", f"[dim]{wall_time}[/]", f"[dim]{cpu_time}[/]")


def _target_count(metrics: dict[str, MetricSummary]) -> int | None:
    if not metrics:
        return None
    return max(metric.count for metric in metrics.values())


def _reconstruction_metric_values(metrics: dict[str, MetricSummary]) -> list[MetricSummary]:
    values = []
    for pattern in (_TOP_K, _ROOT_TOP_K, _GIVEN_ROOT_TOP_K, _PREFIX_TOP_K, _DISTINCT_ROOT_TOP_K):
        values.extend(metric for _, metric, _ in _matching_metrics(metrics, pattern))
    return values


def _solv_tiers(metrics: dict[str, MetricSummary]) -> list[int]:
    tiers: set[int] = set()
    for name in metrics:
        for pattern in (_TIER_VALIDITY, _SOLV_RATE, _MRR_TIER, _MRR_SOLV):
            match = pattern.match(name)
            if match is not None:
                tiers.add(int(match.group(1)))
    return sorted(tiers)


def _solv_labels(metrics: dict[str, MetricSummary], tier: int) -> list[str]:
    labels: set[str] = set()
    for name in metrics:
        solv_match = _SOLV_RATE.match(name)
        if solv_match is not None and int(solv_match.group(1)) == tier:
            labels.add(solv_match.group(2))
        mrr_match = _MRR_SOLV.match(name)
        if mrr_match is not None and int(mrr_match.group(1)) == tier:
            labels.add(mrr_match.group(2))
    return sorted(labels)


def _metric_cell(
    name: str,
    metric: MetricSummary | None,
    *,
    compact_ci: bool = False,
) -> str:
    if metric is None:
        return ""

    value = _format_value(name, metric)
    symbol = _format_reliability_symbol(metric)
    first_line = f"{value}{symbol}" if symbol else value

    ci = _format_compact_ci(name, metric) if compact_ci else _format_ci(name, metric, rich=False)
    if not ci:
        return first_line
    return f"{first_line}\n[dim]{ci}[/]"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds > 24 * 60 * 60:
        return f"{seconds / (24 * 60 * 60):.1f} days"
    if seconds > 60 * 60:
        return f"{seconds / (60 * 60):.1f} hr"
    if seconds > 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.2f} s"


def _format_reliability(metric: MetricSummary, *, rich: bool = True) -> str:
    reliability = metric.reliability
    if reliability is None:
        return ""
    if reliability.code == "OK":
        return "[green]OK[/]" if rich else "OK"
    code = escape(reliability.code) if rich else reliability.code
    return f"[yellow]{code}[/]" if rich else code


def _format_reliability_symbol(metric: MetricSummary) -> str:
    reliability = metric.reliability
    if reliability is None or reliability.code == "OK":
        return ""
    if reliability.code == "EXTREME_P":
        return "[yellow]*[/]"
    return "[yellow]![/]"


def _reliability_legend(metrics: list[MetricSummary]) -> str | None:
    parts = _reliability_legend_parts(metrics)
    if not parts:
        return None
    return "flags: " + "; ".join(f"{symbol} {description}" for symbol, description in parts)


def _reliability_legend_parts(metrics: list[MetricSummary]) -> list[tuple[str, str]]:
    symbols = {_plain_reliability_symbol(metric) for metric in metrics}
    symbols.discard("")
    parts = []
    if "*" in symbols:
        parts.append(("*", "extreme probability"))
    if "!" in symbols:
        parts.append(("!", "low n / unstable ci"))
    return parts


def _plain_reliability_symbol(metric: MetricSummary) -> str:
    reliability = metric.reliability
    if reliability is None or reliability.code == "OK":
        return ""
    if reliability.code == "EXTREME_P":
        return "*"
    return "!"


def _headline_metrics(metrics: dict[str, MetricSummary]) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    ordered = _hierarchy_metrics(metrics)
    ordered.extend(_headline_reconstruction_metrics(metrics))
    return ordered


def _hierarchy_metrics(metrics: dict[str, MetricSummary]) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    matches: list[tuple[str, MetricSummary, re.Match[str]]] = []
    for pattern in (_TIER_VALIDITY, _SOLV_RATE, _MRR_TIER, _MRR_SOLV):
        matches.extend(_matching_metrics(metrics, pattern))
    return sorted(matches, key=_hierarchy_sort_key)


def _headline_reconstruction_metrics(
    metrics: dict[str, MetricSummary],
) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    return _matching_metrics(metrics, _TOP_K)


def _hierarchy_sort_key(item: tuple[str, MetricSummary, re.Match[str]]) -> tuple[int, int, str]:
    name, _, match = item
    pattern = match.re
    if pattern is _TIER_VALIDITY:
        return (int(match.group(1)), 0, "")
    if pattern is _SOLV_RATE:
        return (int(match.group(1)), 1, match.group(2))
    if pattern is _MRR_TIER:
        return (int(match.group(1)), 2, "")
    if pattern is _MRR_SOLV:
        return (int(match.group(1)), 3, match.group(2))
    raise ValueError(f"unexpected hierarchy metric: {name!r}")


def _matching_metrics(
    metrics: dict[str, MetricSummary], pattern: re.Pattern[str]
) -> list[tuple[str, MetricSummary, re.Match[str]]]:
    matches = []
    for name, metric in metrics.items():
        match = pattern.match(name)
        if match is not None:
            matches.append((name, metric, match))
    return sorted(matches, key=lambda item: int(item[2].group(1)))


def _diagnostic_ks(metrics: dict[str, MetricSummary]) -> list[int]:
    ks: set[int] = set()
    for name in metrics:
        for pattern in (_TOP_K, _ROOT_TOP_K, _GIVEN_ROOT_TOP_K, _DISTINCT_ROOT_TOP_K):
            match = pattern.match(name)
            if match is not None:
                ks.add(int(match.group(1)))
        prefix_match = _PREFIX_TOP_K.match(name)
        if prefix_match is not None:
            ks.add(int(prefix_match.group(2)))
    return sorted(ks)


def _diagnostic_prefix_depths(metrics: dict[str, MetricSummary]) -> list[int]:
    depths = {int(match.group(1)) for name in metrics if (match := _PREFIX_TOP_K.match(name)) is not None}
    return sorted(depths)


def _display_metric_name(name: str, match: re.Match[str]) -> str:
    pattern = match.re
    if pattern is _TIER_VALIDITY:
        return _plain_tier_validity_label(match)
    if pattern is _SOLV_RATE:
        return _plain_solv_label(match)
    if pattern is _MRR_TIER:
        return _plain_mrr_tier_label(match)
    if pattern is _MRR_SOLV:
        return _plain_mrr_label(match)
    if pattern is _TOP_K:
        return _plain_top_k_label(match)
    if pattern is _ROOT_TOP_K:
        return _plain_root_top_k_label(match)
    if pattern is _GIVEN_ROOT_TOP_K:
        return _plain_given_root_top_k_label(match)
    if pattern is _PREFIX_TOP_K:
        return _plain_prefix_top_k_label(match)
    if pattern is _DISTINCT_ROOT_TOP_K:
        return _plain_distinct_root_top_k_label(match)
    raise ValueError(f"unexpected report metric: {name!r}")


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


def _plain_root_top_k_label(match: re.Match[str]) -> str:
    return f"Root Top-{match.group(1)}"


def _plain_given_root_top_k_label(match: re.Match[str]) -> str:
    return f"Route given root Top-{match.group(1)}"


def _plain_prefix_top_k_label(match: re.Match[str]) -> str:
    return f"Prefix depth {match.group(1)} Top-{match.group(2)}"


def _plain_distinct_root_top_k_label(match: re.Match[str]) -> str:
    return f"Distinct roots Top-{match.group(1)}"


def _format_value(name: str, metric: MetricSummary, *, rich: bool = True) -> str:
    if name.startswith("mrr_") or name.startswith("distinct_root_reactions_"):
        value = f"{metric.value:.3f}"
    else:
        value = f"{metric.value:.1%}"
    return value


def _format_ci(name: str, metric: MetricSummary, *, rich: bool = True) -> str:
    if metric.ci_low is None or metric.ci_high is None:
        return ""
    if name.startswith("mrr_") or name.startswith("distinct_root_reactions_"):
        value = f"[{metric.ci_low:.3f}, {metric.ci_high:.3f}]"
    else:
        value = f"[{metric.ci_low:.1%}, {metric.ci_high:.1%}]"
    return f"[green]{value}[/]" if rich else value


def _format_compact_ci(name: str, metric: MetricSummary) -> str:
    if metric.ci_low is None or metric.ci_high is None:
        return ""
    if name.startswith("mrr_") or name.startswith("distinct_root_reactions_"):
        return f"[{metric.ci_low:.2f}, {metric.ci_high:.2f}]"
    return f"[{metric.ci_low * 100:.1f}, {metric.ci_high * 100:.1f}]"
