"""
create a master markdown audit for a released paroutes training-set version.

usage:
    uv run scripts/paroutes/training-set-prep/03-audit-release.py
    uv run scripts/paroutes/training-set-prep/03-audit-release.py --release-root path/to/v2026-05-11
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrocast.curation.training import (
    TrainingReactionRecord,
    TrainingRouteRecord,
    adapt_training_routes,
    build_route_release_split_audit,
    render_route_release_split_audit_markdown,
)
from retrocast.curation.training.route_release import release_steps
from retrocast.exceptions import TrainingReleaseError
from retrocast.io import (
    load_raw_paroutes_list,
    load_training_reaction_records,
    load_training_reaction_smiles,
    load_training_route_records,
)
from retrocast.models.chem import ReactionSignature
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
RELEASE_VERSION = "v2026-05-12"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
DEFAULT_OUTPUT_PATH = DEFAULT_RELEASE_ROOT / "release-audit.md"
ROUTE_RELEASE_NAMES = ("route-holdout-n1-n5", "reaction-holdout-n1-n5")
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


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="audit released paroutes training-set splits into one markdown file.")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help=f"release version root. default: {DEFAULT_RELEASE_ROOT}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"markdown report path. default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"raw paroutes asset directory for holdout leak checks. default: {RAW_DIR}",
    )
    args = parser.parse_args()

    release_dirs: dict[str, Path] = {}
    for release_name in ROUTE_RELEASE_NAMES:
        release_dir = args.release_root / release_name
        if missing_files := [path.name for path in required_release_files(release_dir) if not path.exists()]:
            logger.info("skipping %s because route files are missing: %s", release_name, ", ".join(missing_files))
            continue
        release_dirs[release_name] = release_dir

    if not release_dirs:
        raise FileNotFoundError(f"no route releases found under {args.release_root}")

    holdout = load_holdout_reference(args.raw_dir)
    audits = []
    sanity_checks: dict[str, dict[str, int]] = {}
    for release_name, release_dir in release_dirs.items():
        files = load_route_release_files(release_dir)
        records = [*files.training, *files.validation]
        audits.append(build_route_release_split_audit(release_name=release_name, route_records=records))
        sanity_checks[release_name] = audit_route_release_sanity(
            release_name=release_name,
            files=files,
            holdout=holdout,
        )

    report = render_route_release_split_audit_markdown(
        release_root_name=args.release_root.name,
        audits=audits,
    )
    report = f"{report}\n{render_sanity_checks_markdown(sanity_checks)}"
    single_step_checks = audit_single_step_release_if_present(
        release_root=args.release_root,
        parent_route_ids={record.id for record in load_route_release_files(release_dirs["reaction-holdout-n1-n5"]).all}
        if "reaction-holdout-n1-n5" in release_dirs
        else set(),
    )
    if single_step_checks is not None:
        report = f"{report}\n{render_single_step_sanity_markdown(single_step_checks)}"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    logger.info("wrote release audit to %s", args.output_path)


def required_release_files(release_dir: Path) -> tuple[Path, Path, Path]:
    return (
        release_dir / "all.jsonl.gz",
        release_dir / "training.jsonl.gz",
        release_dir / "validation.jsonl.gz",
    )


def load_route_release_files(release_dir: Path) -> RouteReleaseFiles:
    all_path, training_path, validation_path = required_release_files(release_dir)
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
    content_hash_mismatches = 0
    route_signature_mismatches = 0
    route_metadata_mismatches = 0
    for record in files.all:
        if record.content_hash != record.route.get_content_hash():
            content_hash_mismatches += 1
        if record.route_signature != record.route.get_structural_signature():
            route_signature_mismatches += 1
        if not route_metadata_matches_sources(record):
            route_metadata_mismatches += 1
        existing_split = route_signature_by_split.setdefault(record.route_signature, record.split)
        if existing_split != record.split:
            split_overlaps += 1

    failures.update(
        {
            key: value
            for key, value in {
                "content_hash_mismatches": content_hash_mismatches,
                "route_signature_mismatches": route_signature_mismatches,
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


if __name__ == "__main__":
    main()
