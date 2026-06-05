"""Migrate schema v2 benchmark task constraints to serialized constraint records.

Usage:
    uv run scripts/migrations/05-migrate-task-constraints-to-records.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from retrocast.hashing import hash_file, hash_json
from retrocast.io import load_json_gz, save_benchmark
from retrocast.models import (
    Benchmark,
    RequiredLeavesConstraint,
    RouteDepthConstraint,
    StockTerminationConstraint,
    TaskConstraint,
)
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.utils.logging import configure_script_logging, logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS_DIR = PROJECT_ROOT / "data" / "retrocast" / "1-benchmarks" / "definitions"
DATA_ROOT = PROJECT_ROOT / "data" / "retrocast"
BENCHMARK_NAMES = (
    "mkt-cnv-160",
    "mkt-lin-500",
    "paroutes-n1-full-buyables-pruned",
    "paroutes-n1-full-pruned",
    "paroutes-n5-full-buyables-pruned",
    "paroutes-n5-full-pruned",
    "random-n5-100",
    "random-n5-1000",
    "random-n5-2000",
    "random-n5-250",
    "random-n5-50",
    "random-n5-500",
    "ref-cnv-400",
    "ref-lin-600",
    "ref-lng-84",
    "uspto-190",
    "uspto-83",
)


def main() -> None:
    configure_script_logging()
    for benchmark_name in BENCHMARK_NAMES:
        path = DEFINITIONS_DIR / f"{benchmark_name}.json.gz"
        data = load_json_gz(path)
        migrated = migrate_benchmark_payload(data)
        benchmark = Benchmark.model_validate(migrated)
        manifest_path = path.with_name(path.name.replace(".json.gz", ".manifest.json"))
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

        save_benchmark(benchmark, path)
        manifest = migrate_manifest(manifest, benchmark=benchmark, benchmark_path=path)
        manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        logger.info("wrote %s", path.name)


def migrate_benchmark_payload(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    migrated["default_constraints"] = migrate_constraint_set(data["default_constraints"])
    migrated["constraints"] = {
        target_id: constraints
        for target_id, value in cast(dict[str, Any], data["constraints"]).items()
        if (constraints := migrate_constraint_set(value))
    }
    return migrated


def migrate_constraint_set(value: object) -> list[TaskConstraint]:
    legacy_constraints = cast(dict[str, Any], value)
    constraints: list[TaskConstraint] = []

    if "stock" in legacy_constraints:
        constraints.append(StockTerminationConstraint(stock=legacy_constraints["stock"]))

    if "required_leaves_smiles" in legacy_constraints:
        constraints.append(RequiredLeavesConstraint(smiles=legacy_constraints["required_leaves_smiles"]))

    if "route_depth" in legacy_constraints:
        constraints.append(RouteDepthConstraint(max_depth=legacy_constraints["route_depth"]))

    return constraints


def migrate_manifest(
    manifest: Manifest,
    *,
    benchmark: Benchmark,
    benchmark_path: Path,
) -> Manifest:
    benchmark_relpath = str(benchmark_path.resolve().relative_to(DATA_ROOT.resolve()))
    previous_output = manifest.iter_output_files()[0]
    source_files = [
        *manifest.source_files,
        FileInfo(
            path=benchmark_relpath,
            file_hash=previous_output.file_hash,
            content_hash=previous_output.content_hash,
        ),
    ]
    output_files = [
        FileInfo(
            path=benchmark_relpath,
            file_hash=hash_file(benchmark_path),
            content_hash=hash_json(benchmark.model_dump(mode="json", exclude_none=True)),
        )
    ]

    return manifest.model_copy(
        update={
            "schema_version": "2",
            "source_files": source_files,
            "output_files": output_files,
            "statistics": {
                **manifest.statistics,
                "n_targets": len(benchmark.targets),
                "metric_label": benchmark.derived_metric_label(),
            },
            "summary": {
                **manifest.summary,
                "migrations": [
                    *manifest.summary.get("migrations", []),
                    {
                        "action": "scripts/migrations/05-migrate-task-constraints-to-records",
                        "previous_schema_version": manifest.schema_version,
                        "schema_version": benchmark.schema_version,
                    },
                ],
            },
        }
    )


if __name__ == "__main__":
    main()
