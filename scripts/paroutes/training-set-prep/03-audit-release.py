"""
create a master markdown audit for a released paroutes training-set version.

usage:
    uv run scripts/paroutes/training-set-prep/03-audit-release.py
    uv run scripts/paroutes/training-set-prep/03-audit-release.py --release-root path/to/v2026-05-12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from retrocast.cli.progress import step_progress
from retrocast.curation.training.audit import (
    RouteReleaseFiles,
    audit_route_release_sanity,
    audit_single_step_release_if_present,
    build_route_release_split_audit,
    load_route_release_files,
    render_route_release_split_audit_markdown,
    required_route_release_files,
)
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RELEASE_VERSION = "v2026-05-29"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
ROUTE_RELEASE_NAMES = ("route-holdout-n1-n5", "reaction-holdout-n1-n5")
SINGLE_STEP_RELEASE_BY_PARENT = {
    "route-holdout-n1-n5": "single-step-route-holdout-n1-n5",
    "reaction-holdout-n1-n5": "single-step-reaction-holdout-n1-n5",
}


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
        default=None,
        help="markdown report path. default: <release-root>/release-audit.md",
    )
    args = parser.parse_args()
    output_path = args.output_path or args.release_root / "release-audit.md"

    release_dirs = {
        name: args.release_root / name
        for name in ROUTE_RELEASE_NAMES
        if not missing_route_files(args.release_root / name)
    }
    if not release_dirs:
        raise FileNotFoundError(f"no route releases found under {args.release_root}")

    with step_progress(console=Console(), total=audit_step_count(release_dirs), transient=True) as step:
        report = build_audit_report(release_root=args.release_root, release_dirs=release_dirs, step=step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("wrote release audit to %s", output_path)


def build_audit_report(*, release_root: Path, release_dirs: dict[str, Path], step) -> str:
    audits = []
    summary_rows = []
    route_files: dict[str, RouteReleaseFiles] = {}
    for release_name, release_dir in release_dirs.items():
        with step(f"{release_name}: load route files"):
            files = load_route_release_files(release_dir)
        route_files[release_name] = files
        with step(f"{release_name}: build depth audit"):
            audit = build_route_release_split_audit(
                release_name=release_name, route_records=[*files.training, *files.validation]
            )
        audits.append(audit)
        summary_rows.append(audit)
        with step(f"{release_name}: sanity checks"):
            audit_route_release_sanity(release_name=release_name, files=files)

    for parent_release_name, single_step_release_name in SINGLE_STEP_RELEASE_BY_PARENT.items():
        if parent_release_name not in release_dirs:
            continue
        parent_route_ids = {record.id for record in route_files[parent_release_name].all}
        with step(f"{single_step_release_name}: sanity checks"):
            single_step_checks = audit_single_step_release_if_present(
                release_root=release_root,
                parent_route_ids=parent_route_ids,
                release_name=single_step_release_name,
            )
        if single_step_checks is not None:
            summary_rows.append(single_step_checks)
    return render_route_release_split_audit_markdown(
        release_root_name=release_root.name,
        audits=audits,
        summary_rows=summary_rows,
    )


def audit_step_count(release_dirs: dict[str, Path]) -> int:
    route_steps = 3 * len(release_dirs)
    single_step_steps = sum(
        parent_release_name in release_dirs for parent_release_name in SINGLE_STEP_RELEASE_BY_PARENT
    )
    return route_steps + single_step_steps


def missing_route_files(release_dir: Path) -> list[Path]:
    missing = [path for path in required_route_release_files(release_dir) if not path.exists()]
    if missing:
        logger.info(
            "skipping %s because route files are missing: %s",
            release_dir.name,
            ", ".join(path.name for path in missing),
        )
    return missing


if __name__ == "__main__":
    main()
