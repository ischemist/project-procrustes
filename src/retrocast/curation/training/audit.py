from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import TrainingReactionRecord
from retrocast.curation.training.route_release import _route_transform_key
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import iter_jsonl_gz, load_lines_gz, load_raw_paroutes_list, load_training_route_records


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
        all=load_training_route_records(release_dir / "all.jsonl.gz"),
        training=load_training_route_records(release_dir / "training.jsonl.gz"),
        validation=load_training_route_records(release_dir / "validation.jsonl.gz"),
    )


def build_route_release_split_audit(*, release_name: str, route_records: list[Any]) -> dict[str, Any]:
    by_depth: Counter[int] = Counter()
    by_convergence: Counter[bool] = Counter()
    by_split: Counter[str] = Counter()
    by_depth_split: Counter[tuple[int, str]] = Counter()
    by_convergence_split: Counter[tuple[bool, str]] = Counter()
    for record in route_records:
        depth = record.route.depth()
        convergent = record.route.is_convergent()
        by_depth[depth] += 1
        by_convergence[convergent] += 1
        by_split[record.split] += 1
        by_depth_split[(depth, record.split)] += 1
        by_convergence_split[(convergent, record.split)] += 1
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
        "by_convergence": [
            {
                "convergent": convergent,
                "total": by_convergence[convergent],
                "training": by_convergence_split[(convergent, "training")],
                "validation": by_convergence_split[(convergent, "validation")],
            }
            for convergent in (False, True)
        ],
    }


def render_route_release_split_audit_markdown(*, release_root_name: str, audits: list[dict[str, Any]]) -> str:
    lines = [f"# training release audit: {release_root_name}", "", "| release | total | training | validation | val% |"]
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for audit in audits:
        lines.append(
            f"| {audit['release_name']} | {audit['total']} | {audit['training']} | {audit['validation']} | {_format_percent(_fraction(audit['validation'], audit['total']))} |"
        )
    lines.append("")
    for audit in audits:
        lines.extend(
            [
                f"## {audit['release_name']}",
                "",
                "| depth | total | training | validation |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in audit["by_depth"]:
            lines.append(f"| {row['depth']} | {row['total']} | {row['training']} | {row['validation']} |")
        lines.extend(["", "| convergent | total | training | validation |", "| --- | ---: | ---: | ---: |"])
        for row in audit["by_convergence"]:
            lines.append(
                f"| {_format_bool(row['convergent'])} | {row['total']} | {row['training']} | {row['validation']} |"
            )
        lines.append("")
    return "\n".join(lines)


def load_holdout_reference(raw_dir: Path) -> dict[str, int]:
    return {
        "n1": len(load_raw_paroutes_list(raw_dir / "n1-routes.json.gz"))
        if (raw_dir / "n1-routes.json.gz").exists()
        else 0,
        "n5": len(load_raw_paroutes_list(raw_dir / "n5-routes.json.gz"))
        if (raw_dir / "n5-routes.json.gz").exists()
        else 0,
    }


def audit_route_release_sanity(
    *, release_name: str, files: RouteReleaseFiles, holdout: dict[str, int]
) -> dict[str, Any]:
    all_by_id = {record.id: record for record in files.all}
    split_records = [*files.training, *files.validation]
    split_by_id = {record.id: record for record in split_records}
    failures: dict[str, Any] = {}
    if len(all_by_id) != len(files.all):
        failures["duplicate_record_ids_in_all"] = len(files.all) - len(all_by_id)
    if len(split_by_id) != len(split_records):
        failures["duplicate_record_ids_in_splits"] = len(split_records) - len(split_by_id)
    if set(all_by_id) != set(split_by_id):
        failures["all_split_id_delta"] = len(set(all_by_id) ^ set(split_by_id))
    if wrong_training := sum(record.split != "training" for record in files.training):
        failures["training_file_wrong_split_records"] = wrong_training
    if wrong_validation := sum(record.split != "validation" for record in files.validation):
        failures["validation_file_wrong_split_records"] = wrong_validation
    if duplicate_count(_route_transform_key(record.route) for record in files.training):
        failures["duplicate_training_route_identities"] = duplicate_count(
            _route_transform_key(record.route) for record in files.training
        )
    if duplicate_count(_route_transform_key(record.route) for record in files.validation):
        failures["duplicate_validation_route_identities"] = duplicate_count(
            _route_transform_key(record.route) for record in files.validation
        )
    if failures:
        raise TrainingReleaseError(
            f"{release_name} failed route release sanity checks",
            code="workflow.route_release_audit_failed",
            context={"release_name": release_name, "failures": failures},
        )
    return {
        "release_name": release_name,
        "all_count": len(files.all),
        "training_count": len(files.training),
        "validation_count": len(files.validation),
        "holdout_count": sum(holdout.values()),
    }


def audit_single_step_release_if_present(*, release_root: Path, parent_route_ids: set[str]) -> dict[str, Any] | None:
    release_dir = release_root / "single-step-reaction-holdout-n1-n5"
    if not (release_dir / "all.jsonl.gz").exists():
        return None
    files = load_single_step_release_files(release_dir)
    all_by_id = {record.id: record for record in files.all}
    split_records = [*files.training, *files.validation]
    split_by_id = {record.id: record for record in split_records}
    failures: dict[str, Any] = {}
    if len(all_by_id) != len(files.all):
        failures["duplicate_record_ids_in_all"] = len(files.all) - len(all_by_id)
    if set(all_by_id) != set(split_by_id):
        failures["all_split_id_delta"] = len(set(all_by_id) ^ set(split_by_id))
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
    if duplicate_count(exact_reaction_record_key(record) for record in files.training):
        failures["duplicate_training_exact_reaction_keys"] = duplicate_count(
            exact_reaction_record_key(record) for record in files.training
        )
    training_identities = {reaction_record_identity_key(record) for record in files.training}
    validation_identities = {reaction_record_identity_key(record) for record in files.validation}
    if shared := training_identities & validation_identities:
        failures["cross_split_reaction_identity_overlap"] = len(shared)
    if missing_sources := sum(
        source.route_id not in parent_route_ids for record in files.all for source in record.sources
    ):
        failures["missing_parent_route_sources"] = missing_sources
    if failures:
        raise TrainingReleaseError(
            "single-step-reaction-holdout-n1-n5 failed single-step sanity checks",
            code="workflow.single_step_release_audit_failed",
            context={"release_name": release_dir.name, "failures": failures},
        )
    return {
        "release_name": release_dir.name,
        "records": len(files.all),
        "training": len(files.training),
        "validation": len(files.validation),
        "parent_routes": len(parent_route_ids),
    }


def render_sanity_checks_markdown(sanity_checks: dict[str, dict[str, Any]]) -> str:
    lines = [
        "## sanity checks",
        "",
        "| release | records | training | validation | holdout refs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, checks in sorted(sanity_checks.items()):
        lines.append(
            f"| {name} | {checks['all_count']} | {checks['training_count']} | {checks['validation_count']} | {checks['holdout_count']} |"
        )
    return "\n".join(lines) + "\n"


def render_single_step_sanity_markdown(checks: dict[str, Any]) -> str:
    return (
        "## single-step sanity\n\n"
        "| release | records | training | validation | parent routes |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"| {checks['release_name']} | {checks['records']} | {checks['training']} | {checks['validation']} | {checks['parent_routes']} |\n"
    )


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


def exact_reaction_record_key(record: TrainingReactionRecord) -> tuple[str, tuple[str, ...], str | None]:
    return (
        str(record.mapped_smiles),
        tuple(str(value) for value in record.condition_slot_smiles),
        record.condition_slot,
    )


def reaction_record_identity_key(
    record: TrainingReactionRecord,
) -> tuple[tuple[str, ...], str, tuple[str, ...] | str | None]:
    return (
        tuple(sorted(str(value) for value in record.reactants)),
        str(record.product),
        tuple(str(value) for value in record.condition_slot_smiles) or record.condition_slot,
    )


def duplicate_count(values) -> int:
    values = list(values)
    return len(values) - len(set(values))


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_percent(value: float) -> str:
    return f"{value:.4%}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"
