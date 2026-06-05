"""
create public paroutes n1/n5 test-set release files.

usage:
    uv run scripts/paroutes/training-set-prep/06-create-test-set-release.py
    uv run scripts/paroutes/training-set-prep/06-create-test-set-release.py --dataset n1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from rich.console import Console

from retrocast.cli.progress import step_progress
from retrocast.curation.training.records import TestSetName
from retrocast.curation.training.route_release import adapt_training_routes
from retrocast.curation.training.testset_release import (
    build_test_reaction_records,
    build_test_route_records,
    write_test_reaction_release,
    write_test_route_release,
)
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
RELEASE_VERSION = "v2026-05-29"
DEFAULT_OUTPUT_DIR = DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION
TEST_DATASETS: tuple[TestSetName, ...] = ("n1", "n5")


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="create paroutes n1/n5 test-set release files.")
    parser.add_argument(
        "--dataset",
        choices=[*TEST_DATASETS, "both"],
        default="both",
        help="test set to release. default: both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory root. default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    datasets = list(TEST_DATASETS) if args.dataset == "both" else [cast(TestSetName, args.dataset)]
    with step_progress(console=Console(), total=4 * len(datasets), transient=True) as step:
        for dataset in datasets:
            build_test_set_release(dataset=dataset, output_dir=args.output_dir, step=step)


def build_test_set_release(*, dataset: TestSetName, output_dir: Path, step) -> None:
    source_path = RAW_DIR / f"{dataset}-routes.json.gz"
    with step(f"{dataset}: adapt routes"):
        adaptation = adapt_training_routes(source_path, dataset=dataset, show_progress=False)
    with step(f"{dataset}: build route records"):
        route_records = build_test_route_records(dataset=dataset, routes=adaptation.routes)
    with step(f"{dataset}: write route release"):
        write_test_route_release(
            dataset=dataset,
            records=route_records,
            adaptation=adaptation.stats,
            output_dir=output_dir,
            source_paths=[source_path],
            source_root=BASE_DIR,
        )

    with step(f"{dataset}: build and write reactions"):
        reaction_records = build_test_reaction_records(dataset=dataset, route_records=route_records)
        route_manifest_path = output_dir / f"{dataset}-routes" / "manifest.json"
        reaction_sources = [output_dir / f"{dataset}-routes" / "all.jsonl.gz"]
        if route_manifest_path.exists():
            reaction_sources.append(route_manifest_path)
        write_test_reaction_release(
            dataset=dataset,
            records=reaction_records,
            output_dir=output_dir,
            source_paths=reaction_sources,
            source_root=BASE_DIR,
        )

    logger.info(
        "wrote %s test release: %s routes, %s single-step reactions.",
        dataset,
        f"{len(route_records):,}",
        f"{len(reaction_records):,}",
    )


if __name__ == "__main__":
    main()
