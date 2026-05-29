"""
create a master markdown audit for a released paroutes training-set version.

usage:
    uv run scripts/paroutes/training-set-prep/03-audit-release.py
    uv run scripts/paroutes/training-set-prep/03-audit-release.py --release-root path/to/v2026-05-12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrocast.curation.training import build_route_release_split_audit, render_route_release_split_audit_markdown
from retrocast.curation.training.audit import (
    audit_route_release_sanity,
    audit_single_step_release_if_present,
    load_holdout_reference,
    load_route_release_files,
    render_sanity_checks_markdown,
    render_single_step_sanity_markdown,
    required_route_release_files,
)
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
RELEASE_VERSION = "v2026-05-29"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
ROUTE_RELEASE_NAMES = ("route-holdout-n1-n5", "reaction-holdout-n1-n5")


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
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"raw paroutes asset directory for holdout leak checks. default: {RAW_DIR}",
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

    holdout = load_holdout_reference(args.raw_dir)
    sanity_checks = {}
    audits = []
    for release_name, release_dir in release_dirs.items():
        files = load_route_release_files(release_dir)
        audits.append(
            build_route_release_split_audit(
                release_name=release_name, route_records=[*files.training, *files.validation]
            )
        )
        sanity_checks[release_name] = audit_route_release_sanity(
            release_name=release_name,
            files=files,
            holdout=holdout,
        )

    report = render_route_release_split_audit_markdown(release_root_name=args.release_root.name, audits=audits)
    report = f"{report}\n{render_sanity_checks_markdown(sanity_checks)}"

    parent_route_ids = (
        {record.id for record in load_route_release_files(release_dirs["reaction-holdout-n1-n5"]).all}
        if "reaction-holdout-n1-n5" in release_dirs
        else set()
    )
    single_step_checks = audit_single_step_release_if_present(
        release_root=args.release_root,
        parent_route_ids=parent_route_ids,
    )
    if single_step_checks is not None:
        report = f"{report}\n{render_single_step_sanity_markdown(single_step_checks)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("wrote release audit to %s", output_path)


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
