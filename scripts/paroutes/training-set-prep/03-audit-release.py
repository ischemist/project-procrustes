"""
create a master markdown audit for a released paroutes training-set version.

usage:
    uv run scripts/paroutes/training-set-prep/03-audit-release.py
    uv run scripts/paroutes/training-set-prep/03-audit-release.py --release-root path/to/v2026-05-11
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrocast.curation.training import (
    build_route_release_split_audit,
    render_route_release_split_audit_markdown,
)
from retrocast.io import load_training_route_records
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RELEASE_VERSION = "v2026-05-11"
DEFAULT_RELEASE_ROOT = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
DEFAULT_OUTPUT_PATH = DEFAULT_RELEASE_ROOT / "release-audit.md"
ROUTE_RELEASE_NAMES = ("route-heldout-n1-n5", "reaction-heldout-n1-n5")


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
    args = parser.parse_args()

    audits = []
    for release_name in ROUTE_RELEASE_NAMES:
        release_dir = args.release_root / release_name
        training_path = release_dir / "training.jsonl.gz"
        validation_path = release_dir / "validation.jsonl.gz"
        if not training_path.exists() or not validation_path.exists():
            logger.info("skipping %s because training/validation route files are missing", release_name)
            continue

        records = [
            *load_training_route_records(training_path),
            *load_training_route_records(validation_path),
        ]
        audits.append(build_route_release_split_audit(release_name=release_name, route_records=records))

    if not audits:
        raise FileNotFoundError(f"no route releases found under {args.release_root}")

    report = render_route_release_split_audit_markdown(
        release_root_name=args.release_root.name,
        audits=audits,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    logger.info("wrote release audit to %s", args.output_path)


if __name__ == "__main__":
    main()
