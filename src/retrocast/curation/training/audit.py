from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from retrocast.curation.training.records import TrainingRouteRecord


@dataclass(frozen=True)
class SplitAuditCounts:
    training: int
    validation: int

    @property
    def total(self) -> int:
        return self.training + self.validation

    @property
    def validation_fraction(self) -> float:
        if self.total == 0:
            return 0.0
        return self.validation / self.total

    def all_fraction(self, grand_total: int) -> float:
        if grand_total == 0:
            return 0.0
        return self.total / grand_total


@dataclass(frozen=True)
class RouteLengthSplitAuditRow:
    length: int
    counts: SplitAuditCounts


@dataclass(frozen=True)
class RouteConvergenceSplitAuditRow:
    has_convergent_reaction: bool
    counts: SplitAuditCounts


@dataclass(frozen=True)
class RouteLengthConvergenceSplitAuditRow:
    length: int
    has_convergent_reaction: bool
    counts: SplitAuditCounts


@dataclass(frozen=True)
class RouteReleaseSplitAudit:
    release_name: str
    total_counts: SplitAuditCounts
    by_length: tuple[RouteLengthSplitAuditRow, ...]
    by_convergence: tuple[RouteConvergenceSplitAuditRow, ...]
    by_length_and_convergence: tuple[RouteLengthConvergenceSplitAuditRow, ...]


def build_route_release_split_audit(
    *,
    release_name: str,
    route_records: Sequence[TrainingRouteRecord],
) -> RouteReleaseSplitAudit:
    training_length_counts: Counter[int] = Counter()
    validation_length_counts: Counter[int] = Counter()
    training_convergence_counts: Counter[bool] = Counter()
    validation_convergence_counts: Counter[bool] = Counter()
    training_joint_counts: Counter[tuple[int, bool]] = Counter()
    validation_joint_counts: Counter[tuple[int, bool]] = Counter()

    for record in route_records:
        length = record.route.length
        has_convergent_reaction = record.route.has_convergent_reaction
        if record.split == "training":
            training_length_counts[length] += 1
            training_convergence_counts[has_convergent_reaction] += 1
            training_joint_counts[(length, has_convergent_reaction)] += 1
        elif record.split == "validation":
            validation_length_counts[length] += 1
            validation_convergence_counts[has_convergent_reaction] += 1
            validation_joint_counts[(length, has_convergent_reaction)] += 1
        else:
            raise ValueError(f"unexpected split on training route record: {record.split}")

    total_counts = SplitAuditCounts(
        training=sum(training_length_counts.values()),
        validation=sum(validation_length_counts.values()),
    )
    by_length = tuple(
        RouteLengthSplitAuditRow(
            length=length,
            counts=SplitAuditCounts(
                training=training_length_counts[length],
                validation=validation_length_counts[length],
            ),
        )
        for length in sorted(set(training_length_counts) | set(validation_length_counts))
    )
    by_convergence = tuple(
        RouteConvergenceSplitAuditRow(
            has_convergent_reaction=has_convergent_reaction,
            counts=SplitAuditCounts(
                training=training_convergence_counts[has_convergent_reaction],
                validation=validation_convergence_counts[has_convergent_reaction],
            ),
        )
        for has_convergent_reaction in (False, True)
    )
    by_length_and_convergence = tuple(
        RouteLengthConvergenceSplitAuditRow(
            length=length,
            has_convergent_reaction=has_convergent_reaction,
            counts=SplitAuditCounts(
                training=training_joint_counts[(length, has_convergent_reaction)],
                validation=validation_joint_counts[(length, has_convergent_reaction)],
            ),
        )
        for length, has_convergent_reaction in sorted(set(training_joint_counts) | set(validation_joint_counts))
    )

    return RouteReleaseSplitAudit(
        release_name=release_name,
        total_counts=total_counts,
        by_length=by_length,
        by_convergence=by_convergence,
        by_length_and_convergence=by_length_and_convergence,
    )


def render_route_release_split_audit_markdown(
    *,
    release_root_name: str,
    audits: Sequence[RouteReleaseSplitAudit],
) -> str:
    def _format_percent(value: float) -> str:
        return f"{value:.4%}"

    def _format_bool(value: bool) -> str:
        return "true" if value else "false"

    lines = [
        "# release audit",
        "",
        f"release root: `{release_root_name}`",
        "",
        "this report summarizes `training` / `validation` split balance for the released route artifacts.",
        "",
    ]

    for audit in audits:
        lines.extend(
            [
                f"## `{audit.release_name}`",
                "",
                f"totals: `{audit.total_counts.training:,}` training, `{audit.total_counts.validation:,}` validation, "
                f"`{audit.total_counts.total:,}` overall, validation fraction `{_format_percent(audit.total_counts.validation_fraction)}`.",
                "",
                "| convergent | train | val | all | val% | all% |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in audit.by_convergence:
            lines.append(
                f"| `{_format_bool(row.has_convergent_reaction)}` | "
                f"{row.counts.training:,} | {row.counts.validation:,} | {row.counts.total:,} | "
                f"{_format_percent(row.counts.validation_fraction)} | {_format_percent(row.counts.all_fraction(audit.total_counts.total))} |"
            )

        lines.extend(
            [
                "",
                "| length | train | val | all | val% |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in audit.by_length:
            lines.append(
                f"| {row.length} | {row.counts.training:,} | {row.counts.validation:,} | {row.counts.total:,} | "
                f"{_format_percent(row.counts.validation_fraction)} |"
            )

        lines.extend(
            [
                "",
                "| length | non-conv train | non-conv val | non-conv all | non-conv val% | conv train | conv val | conv all | conv val% |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        rows_by_length_and_convergence = {
            (row.length, row.has_convergent_reaction): row for row in audit.by_length_and_convergence
        }
        for length_row in audit.by_length:
            non_convergent_row = rows_by_length_and_convergence.get((length_row.length, False))
            convergent_row = rows_by_length_and_convergence.get((length_row.length, True))

            def _format_side(row: RouteLengthConvergenceSplitAuditRow | None) -> str:
                if row is None:
                    return "0 | 0 | 0 | 0.0000%"
                return (
                    f"{row.counts.training:,} | {row.counts.validation:,} | {row.counts.total:,} | "
                    f"{_format_percent(row.counts.validation_fraction)}"
                )

            lines.append(
                f"| {length_row.length} | {_format_side(non_convergent_row)} | {_format_side(convergent_row)} |"
            )
        lines.append("")

    return "\n".join(lines)
