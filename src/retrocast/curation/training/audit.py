from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import TrainingReactionRecord, TrainingRouteRecord
from retrocast.curation.training.route_release import adapt_training_routes, release_steps
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import (
    load_raw_paroutes_list,
    load_training_reaction_records,
    load_training_reaction_smiles,
    load_training_route_records,
)
from retrocast.models.chem import ReactionSignature
from retrocast.utils.logging import logger

SINGLE_STEP_RELEASE_NAME = "single-step-reaction-holdout-n1-n5"


@dataclass(frozen=True)
class RouteReleaseFiles:
    all: list[TrainingRouteRecord]
    training: list[TrainingRouteRecord]
    validation: list[TrainingRouteRecord]


@dataclass(frozen=True)
class HoldoutReference:
    route_signatures: set[str]
    reaction_signatures: set[ReactionSignature]


@dataclass(frozen=True)
class SingleStepReleaseFiles:
    all: list[TrainingReactionRecord]
    training: list[TrainingReactionRecord]
    validation: list[TrainingReactionRecord]
    all_rsmi_count: int
    training_rsmi_count: int
    validation_rsmi_count: int


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


def required_route_release_files(release_dir: Path) -> tuple[Path, Path, Path]:
    return (
        release_dir / "all.jsonl.gz",
        release_dir / "training.jsonl.gz",
        release_dir / "validation.jsonl.gz",
    )


def load_route_release_files(release_dir: Path) -> RouteReleaseFiles:
    all_path, training_path, validation_path = required_route_release_files(release_dir)
    return RouteReleaseFiles(
        all=load_training_route_records(all_path),
        training=load_training_route_records(training_path),
        validation=load_training_route_records(validation_path),
    )


def load_single_step_release_files(release_dir: Path) -> SingleStepReleaseFiles:
    return SingleStepReleaseFiles(
        all=load_training_reaction_records(release_dir / "all.jsonl.gz"),
        training=load_training_reaction_records(release_dir / "training.jsonl.gz"),
        validation=load_training_reaction_records(release_dir / "validation.jsonl.gz"),
        all_rsmi_count=len(load_training_reaction_smiles(release_dir / "all.rsmi.txt.gz")),
        training_rsmi_count=len(load_training_reaction_smiles(release_dir / "training.rsmi.txt.gz")),
        validation_rsmi_count=len(load_training_reaction_smiles(release_dir / "validation.rsmi.txt.gz")),
    )


def load_holdout_reference(raw_dir: Path) -> HoldoutReference:
    routes = []
    for dataset in ("n1", "n5"):
        raw_routes = load_raw_paroutes_list(raw_dir / f"{dataset}-routes.json.gz")
        adapted_routes, _ = adapt_training_routes(
            raw_routes,
            dataset=dataset,
            id_width=5,
            collect_reactions=True,
            show_progress=False,
        )
        routes.extend(adapted_routes)
    return HoldoutReference(
        route_signatures={route.structural_signature for route in routes},
        reaction_signatures={signature for route in routes for signature in route.reaction_signatures},
    )


def audit_route_release_sanity(
    *,
    release_name: str,
    files: RouteReleaseFiles,
    holdout: HoldoutReference,
) -> dict[str, int]:
    all_records_by_id = {record.id: record for record in files.all}
    split_records = [*files.training, *files.validation]
    split_records_by_id = {record.id: record for record in split_records}

    failures: dict[str, Any] = {}
    if len(all_records_by_id) != len(files.all):
        failures["duplicate_record_ids_in_all"] = len(files.all) - len(all_records_by_id)
    if len(split_records_by_id) != len(split_records):
        failures["duplicate_record_ids_in_splits"] = len(split_records) - len(split_records_by_id)
    if set(all_records_by_id) != set(split_records_by_id):
        failures["all_split_id_delta"] = len(set(all_records_by_id) ^ set(split_records_by_id))
    if any(record.split != "training" for record in files.training):
        failures["training_file_wrong_split_records"] = sum(record.split != "training" for record in files.training)
    if any(record.split != "validation" for record in files.validation):
        failures["validation_file_wrong_split_records"] = sum(
            record.split != "validation" for record in files.validation
        )

    route_signature_by_split: dict[str, str] = {}
    split_overlaps = 0
    route_metadata_mismatches = 0
    for record in files.all:
        if not route_metadata_matches_sources(record):
            route_metadata_mismatches += 1
        existing_split = route_signature_by_split.setdefault(record.route_signature, record.split)
        if existing_split != record.split:
            split_overlaps += 1

    failures.update(
        {
            key: value
            for key, value in {
                "route_metadata_mismatches": route_metadata_mismatches,
                "cross_split_route_signature_overlaps": split_overlaps,
            }.items()
            if value
        }
    )

    exact_keys = [record.route.get_annotated_signature(include_mapped_smiles=True) for record in files.all]
    transform_keys = [
        (record.route_signature, tuple(step.condition_identity for step in release_steps(record.route)))
        for record in files.all
    ]
    duplicate_exact_keys = len(exact_keys) - len(set(exact_keys))
    duplicate_transform_keys = len(transform_keys) - len(set(transform_keys))
    if duplicate_exact_keys:
        failures["duplicate_exact_route_keys"] = duplicate_exact_keys
    if duplicate_transform_keys:
        failures["duplicate_transform_route_keys"] = duplicate_transform_keys

    route_holdout_leaks = sum(record.route_signature in holdout.route_signatures for record in files.all)
    if route_holdout_leaks:
        failures["route_holdout_leaks"] = route_holdout_leaks

    reaction_holdout_leaks = 0
    if release_name == "reaction-holdout-n1-n5":
        reaction_holdout_leaks = sum(
            bool(record.route.get_reaction_signatures() & holdout.reaction_signatures) for record in files.all
        )
        if reaction_holdout_leaks:
            failures["reaction_holdout_leaks"] = reaction_holdout_leaks

    if failures:
        raise TrainingReleaseError(
            f"{release_name} failed route release sanity checks",
            code="workflow.route_release_audit_failed",
            context={"release_name": release_name, "failures": failures},
        )

    return {
        "records": len(files.all),
        "training": len(files.training),
        "validation": len(files.validation),
        "duplicate_exact_route_keys": duplicate_exact_keys,
        "duplicate_transform_route_keys": duplicate_transform_keys,
        "route_holdout_leaks": route_holdout_leaks,
        "reaction_holdout_leaks": reaction_holdout_leaks,
    }


def route_metadata_matches_sources(record: TrainingRouteRecord) -> bool:
    if "patent_id" in record.route.metadata:
        return False
    patent_ids = sorted({source.patent_id for source in record.sources if source.patent_id is not None})
    if patent_ids:
        return record.route.metadata.get("source_patent_ids") == patent_ids
    return "source_patent_ids" not in record.route.metadata


def audit_single_step_release_if_present(
    *,
    release_root: Path,
    parent_route_ids: set[str],
) -> dict[str, int] | None:
    release_dir = release_root / SINGLE_STEP_RELEASE_NAME
    required_files = (
        release_dir / "all.jsonl.gz",
        release_dir / "training.jsonl.gz",
        release_dir / "validation.jsonl.gz",
        release_dir / "all.rsmi.txt.gz",
        release_dir / "training.rsmi.txt.gz",
        release_dir / "validation.rsmi.txt.gz",
    )
    if missing_files := [path.name for path in required_files if not path.exists()]:
        logger.info(
            "skipping %s because reaction files are missing: %s",
            SINGLE_STEP_RELEASE_NAME,
            ", ".join(missing_files),
        )
        return None

    files = load_single_step_release_files(release_dir)
    all_records_by_id = {record.id: record for record in files.all}
    split_records = [*files.training, *files.validation]
    split_records_by_id = {record.id: record for record in split_records}
    failures: dict[str, Any] = {}

    if len(all_records_by_id) != len(files.all):
        failures["duplicate_record_ids_in_all"] = len(files.all) - len(all_records_by_id)
    if len(split_records_by_id) != len(split_records):
        failures["duplicate_record_ids_in_splits"] = len(split_records) - len(split_records_by_id)
    if set(all_records_by_id) != set(split_records_by_id):
        failures["all_split_id_delta"] = len(set(all_records_by_id) ^ set(split_records_by_id))
    if any(record.split != "training" for record in files.training):
        failures["training_file_wrong_split_records"] = sum(record.split != "training" for record in files.training)
    if any(record.split != "validation" for record in files.validation):
        failures["validation_file_wrong_split_records"] = sum(
            record.split != "validation" for record in files.validation
        )

    rsmi_count_mismatches = {
        "all": len(files.all) - files.all_rsmi_count,
        "training": len(files.training) - files.training_rsmi_count,
        "validation": len(files.validation) - files.validation_rsmi_count,
    }
    if any(rsmi_count_mismatches.values()):
        failures["rsmi_count_mismatches"] = rsmi_count_mismatches

    duplicate_training_exact_keys = duplicate_count(exact_reaction_record_key(record) for record in files.training)
    duplicate_validation_exact_keys = duplicate_count(exact_reaction_record_key(record) for record in files.validation)
    training_identity_keys = {reaction_record_identity_key(record) for record in files.training}
    validation_identity_keys = {reaction_record_identity_key(record) for record in files.validation}
    shared_identity_keys = training_identity_keys & validation_identity_keys
    missing_parent_route_sources = sum(
        source.route_id not in parent_route_ids for record in files.all for source in record.sources
    )

    if duplicate_training_exact_keys:
        failures["duplicate_training_exact_reaction_keys"] = duplicate_training_exact_keys
    if duplicate_validation_exact_keys:
        failures["duplicate_validation_exact_reaction_keys"] = duplicate_validation_exact_keys
    if shared_identity_keys:
        failures["cross_split_reaction_identity_overlap"] = len(shared_identity_keys)
    if missing_parent_route_sources:
        failures["missing_parent_route_sources"] = missing_parent_route_sources

    if failures:
        raise TrainingReleaseError(
            f"{SINGLE_STEP_RELEASE_NAME} failed single-step sanity checks",
            code="workflow.single_step_release_audit_failed",
            context={"release_name": SINGLE_STEP_RELEASE_NAME, "failures": failures},
        )

    return {
        "records": len(files.all),
        "training": len(files.training),
        "validation": len(files.validation),
        "training_exact_dupes": duplicate_training_exact_keys,
        "validation_exact_dupes": duplicate_validation_exact_keys,
        "cross_split_identity_overlap": len(shared_identity_keys),
        "missing_parent_route_sources": missing_parent_route_sources,
    }


def exact_reaction_record_key(record: TrainingReactionRecord) -> tuple[str, tuple[str, ...], str | None]:
    return (record.mapped_smiles, tuple(record.condition_slot_smiles), record.condition_slot)


def reaction_record_identity_key(
    record: TrainingReactionRecord,
) -> tuple[tuple[str, ...], str, tuple[str, ...] | str | None]:
    return (tuple(record.reactants), record.product, tuple(record.condition_slot_smiles) or record.condition_slot)


def duplicate_count(values: Iterable[Any]) -> int:
    values = list(values)
    return len(values) - len(set(values))


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
            raise TrainingReleaseError(
                f"unexpected split on training route record: {record.split}",
                code="workflow.training_release_unexpected_split",
                context={"record_id": record.id, "split": record.split},
            )

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
                f"`{audit.total_counts.total:,}` overall, validation fraction `{format_percent(audit.total_counts.validation_fraction)}`.",
                "",
                "| convergent | train | val | all | val% | all% |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in audit.by_convergence:
            lines.append(
                f"| `{format_bool(row.has_convergent_reaction)}` | "
                f"{row.counts.training:,} | {row.counts.validation:,} | {row.counts.total:,} | "
                f"{format_percent(row.counts.validation_fraction)} | {format_percent(row.counts.all_fraction(audit.total_counts.total))} |"
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
                f"{format_percent(row.counts.validation_fraction)} |"
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
            lines.append(
                f"| {length_row.length} | {format_length_convergence_side(non_convergent_row)} | "
                f"{format_length_convergence_side(convergent_row)} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_sanity_checks_markdown(sanity_checks: dict[str, dict[str, int]]) -> str:
    lines = [
        "## sanity checks",
        "",
        "| release | records | training | validation | exact dupes | transform dupes | route leaks | reaction leaks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for release_name, checks in sanity_checks.items():
        lines.append(
            f"| `{release_name}` | {checks['records']:,} | {checks['training']:,} | {checks['validation']:,} | "
            f"{checks['duplicate_exact_route_keys']:,} | {checks['duplicate_transform_route_keys']:,} | "
            f"{checks['route_holdout_leaks']:,} | {checks['reaction_holdout_leaks']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_single_step_sanity_markdown(checks: dict[str, int]) -> str:
    lines = [
        "## single-step sanity checks",
        "",
        "| records | training | validation | training exact dupes | validation exact dupes | split identity overlap | missing route sources |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {checks['records']:,} | {checks['training']:,} | {checks['validation']:,} | "
            f"{checks['training_exact_dupes']:,} | {checks['validation_exact_dupes']:,} | "
            f"{checks['cross_split_identity_overlap']:,} | {checks['missing_parent_route_sources']:,} |"
        ),
        "",
    ]
    return "\n".join(lines)


def format_percent(value: float) -> str:
    return f"{value:.4%}"


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_length_convergence_side(row: RouteLengthConvergenceSplitAuditRow | None) -> str:
    if row is None:
        return "0 | 0 | 0 | 0.0000%"
    return (
        f"{row.counts.training:,} | {row.counts.validation:,} | {row.counts.total:,} | "
        f"{format_percent(row.counts.validation_fraction)}"
    )
