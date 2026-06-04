from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import TrainingReactionRecord
from retrocast.curation.training.route_release import _route_transform_key
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import iter_jsonl_gz, load_lines_gz
from retrocast.markdown import format_integer, markdown_table


@dataclass(frozen=True)
class RouteReleaseFiles:
    all: list[Any]
    training: list[Any]
    validation: list[Any]


@dataclass(frozen=True)
class SingleStepReleaseFiles:
    all: list[TrainingReactionRecord]
    training: list[TrainingReactionRecord]
    validation: list[TrainingReactionRecord]
    all_rsmi_count: int
    training_rsmi_count: int
    validation_rsmi_count: int


def required_route_release_files(release_dir: Path) -> list[Path]:
    return [release_dir / "all.jsonl.gz", release_dir / "training.jsonl.gz", release_dir / "validation.jsonl.gz"]


def load_route_release_files(release_dir: Path) -> RouteReleaseFiles:
    return RouteReleaseFiles(
        all=_load_training_route_records(release_dir / "all.jsonl.gz"),
        training=_load_training_route_records(release_dir / "training.jsonl.gz"),
        validation=_load_training_route_records(release_dir / "validation.jsonl.gz"),
    )


def build_route_release_split_audit(*, release_name: str, route_records: list[Any]) -> dict[str, Any]:
    by_depth: Counter[int] = Counter()
    by_split: Counter[str] = Counter()
    by_depth_split: Counter[tuple[int, str]] = Counter()
    for record in route_records:
        depth = record.route.depth()
        by_depth[depth] += 1
        by_split[record.split] += 1
        by_depth_split[(depth, record.split)] += 1
    return {
        "release_name": release_name,
        "total": len(route_records),
        "training": by_split["training"],
        "validation": by_split["validation"],
        "by_depth": [
            {
                "depth": depth,
                "total": by_depth[depth],
                "training": by_depth_split[(depth, "training")],
                "validation": by_depth_split[(depth, "validation")],
            }
            for depth in sorted(by_depth)
        ],
    }


def render_route_release_split_audit_markdown(
    *, release_root_name: str, audits: list[dict[str, Any]], summary_rows: list[dict[str, Any]] | None = None
) -> str:
    summary = summary_rows if summary_rows is not None else audits
    lines = [
        f"# training release audit: {release_root_name}",
        "",
        markdown_table(
            ["release", "total", "training", "validation", "val%"],
            [
                (
                    row["release_name"],
                    format_integer(row["total"]),
                    format_integer(row["training"]),
                    format_integer(row["validation"]),
                    _format_percent(_fraction(row["validation"], row["total"])),
                )
                for row in summary
            ],
            align=["left", "right", "right", "right", "right"],
        ),
    ]
    lines.append("")
    for audit in audits:
        lines.extend(
            [
                f"## {audit['release_name']}",
                "",
                markdown_table(
                    ["depth", "total", "training", "validation"],
                    [
                        (
                            row["depth"],
                            format_integer(row["total"]),
                            format_integer(row["training"]),
                            format_integer(row["validation"]),
                        )
                        for row in audit["by_depth"]
                    ],
                    align=["right", "right", "right", "right"],
                ),
            ]
        )
        lines.append("")
    return "\n".join(lines)


def audit_route_release_sanity(*, release_name: str, files: RouteReleaseFiles) -> None:
    failures = _split_file_failures(files)
    training_route_identities = [_route_transform_key(record.route) for record in files.training]
    if duplicate_training_route_identities := duplicate_count(training_route_identities):
        failures["duplicate_training_route_identities"] = duplicate_training_route_identities
    validation_route_identities = [_route_transform_key(record.route) for record in files.validation]
    if duplicate_validation_route_identities := duplicate_count(validation_route_identities):
        failures["duplicate_validation_route_identities"] = duplicate_validation_route_identities
    if failures:
        raise TrainingReleaseError(
            f"{release_name} failed route release sanity checks",
            code="workflow.route_release_audit_failed",
            context={"release_name": release_name, "failures": failures},
        )


def audit_single_step_release_if_present(
    *,
    release_root: Path,
    parent_route_ids: set[str],
    release_name: str = "single-step-reaction-holdout-n1-n5",
) -> dict[str, Any] | None:
    release_dir = release_root / release_name
    if not (release_dir / "all.jsonl.gz").exists():
        return None
    allow_cross_split_overlap = release_name == "single-step-route-holdout-n1-n5"
    files = load_single_step_release_files(release_dir)
    failures = _split_file_failures(files, check_split_duplicate_ids=False)
    rsmi_count_mismatches = {
        "all": len(files.all) - files.all_rsmi_count,
        "training": len(files.training) - files.training_rsmi_count,
        "validation": len(files.validation) - files.validation_rsmi_count,
    }
    if any(rsmi_count_mismatches.values()):
        failures["rsmi_count_mismatches"] = rsmi_count_mismatches
    training_exact_reaction_keys = [exact_reaction_record_key(record) for record in files.training]
    if duplicate_training_exact_reaction_keys := duplicate_count(training_exact_reaction_keys):
        failures["duplicate_training_exact_reaction_keys"] = duplicate_training_exact_reaction_keys
    validation_exact_reaction_keys = [exact_reaction_record_key(record) for record in files.validation]
    if duplicate_validation_exact_reaction_keys := duplicate_count(validation_exact_reaction_keys):
        failures["duplicate_validation_exact_reaction_keys"] = duplicate_validation_exact_reaction_keys
    training_identities = {reaction_record_identity_key(record) for record in files.training}
    validation_identities = {reaction_record_identity_key(record) for record in files.validation}
    shared = training_identities & validation_identities
    if shared and not allow_cross_split_overlap:
        failures["cross_split_reaction_identity_overlap"] = len(shared)
    if missing_sources := sum(
        source.route_id not in parent_route_ids for record in files.all for source in record.sources
    ):
        failures["missing_parent_route_sources"] = missing_sources
    if failures:
        raise TrainingReleaseError(
            f"{release_name} failed single-step sanity checks",
            code="workflow.single_step_release_audit_failed",
            context={"release_name": release_dir.name, "failures": failures},
        )
    return {
        "release_name": release_dir.name,
        "total": len(files.all),
        "training": len(files.training),
        "validation": len(files.validation),
        "parent_routes": len(parent_route_ids),
        "cross_split_reaction_identity_overlap": len(shared),
    }


def load_single_step_release_files(release_dir: Path) -> SingleStepReleaseFiles:
    return SingleStepReleaseFiles(
        all=[TrainingReactionRecord.model_validate(row) for row in iter_jsonl_gz(release_dir / "all.jsonl.gz")],
        training=[
            TrainingReactionRecord.model_validate(row) for row in iter_jsonl_gz(release_dir / "training.jsonl.gz")
        ],
        validation=[
            TrainingReactionRecord.model_validate(row) for row in iter_jsonl_gz(release_dir / "validation.jsonl.gz")
        ],
        all_rsmi_count=len(load_lines_gz(release_dir / "all.rsmi.txt.gz")),
        training_rsmi_count=len(load_lines_gz(release_dir / "training.rsmi.txt.gz")),
        validation_rsmi_count=len(load_lines_gz(release_dir / "validation.rsmi.txt.gz")),
    )


def _load_training_route_records(path: Path) -> list[Any]:
    from retrocast.curation.training.records import TrainingRouteRecord

    return [TrainingRouteRecord.model_validate(row) for row in iter_jsonl_gz(path)]


def _split_file_failures(files, *, check_split_duplicate_ids: bool = True) -> dict[str, Any]:
    all_by_id = {record.id: record for record in files.all}
    split_records = [*files.training, *files.validation]
    split_by_id = {record.id: record for record in split_records}
    failures: dict[str, Any] = {}
    if len(all_by_id) != len(files.all):
        failures["duplicate_record_ids_in_all"] = len(files.all) - len(all_by_id)
    if check_split_duplicate_ids and len(split_by_id) != len(split_records):
        failures["duplicate_record_ids_in_splits"] = len(split_records) - len(split_by_id)
    if set(all_by_id) != set(split_by_id):
        failures["all_split_id_delta"] = len(set(all_by_id) ^ set(split_by_id))
    if wrong_training := sum(record.split != "training" for record in files.training):
        failures["training_file_wrong_split_records"] = wrong_training
    if wrong_validation := sum(record.split != "validation" for record in files.validation):
        failures["validation_file_wrong_split_records"] = wrong_validation
    return failures


def exact_reaction_record_key(record: TrainingReactionRecord) -> tuple[str, tuple[str, ...], str | None]:
    return (
        str(record.mapped_smiles),
        tuple(str(value) for value in record.condition_slot_smiles),
        record.condition_slot,
    )


def reaction_record_identity_key(
    record: TrainingReactionRecord,
) -> tuple[tuple[str, ...], str, tuple[str, ...] | str | None]:
    condition_smiles = tuple(str(value) for value in record.condition_slot_smiles)
    condition_key: tuple[str, ...] | str | None
    if condition_smiles:
        condition_key = condition_smiles
    else:
        condition_key = record.condition_slot

    return (
        tuple(sorted(str(value) for value in record.reactants)),
        str(record.product),
        condition_key,
    )


def duplicate_count(values) -> int:
    values = list(values)
    return len(values) - len(set(values))


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_percent(value: float) -> str:
    return f"{value:.4%}"
