"""Migrate v0.5.0 benchmark definition artifacts to schema v2.

By default this script only validates conversion. Pass ``--write`` to move each
v0.5.0 benchmark and manifest into ``definitions/v0.5.0/`` and write schema v2
benchmarks plus refreshed manifests at the original paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from retrocast.io import load_json_gz
from retrocast.io.provenance import calculate_file_hash
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.v2.io import save_benchmark
from retrocast.v2.models import Benchmark, Molecule, Reaction, Route, Target, TaskConstraints

LEGACY_DIRNAME = "v0.5.0"


def main() -> None:
    configure_script_logging()
    args = parse_args()
    definitions_dir = args.definitions_dir
    legacy_dir = definitions_dir / LEGACY_DIRNAME

    benchmark_names = {path.name for path in definitions_dir.glob("*.json.gz") if ".rc0.6." not in path.name} | {
        path.name for path in legacy_dir.glob("*.json.gz")
    }
    benchmark_files = [definitions_dir / name for name in sorted(benchmark_names)]
    logger.info("Found %s benchmark definitions in %s", len(benchmark_files), definitions_dir)

    for benchmark_path in benchmark_files:
        sibling_legacy_path = legacy_sibling_path(benchmark_path)
        if (legacy_dir / benchmark_path.name).exists():
            source_path = legacy_dir / benchmark_path.name
        elif sibling_legacy_path.exists():
            source_path = sibling_legacy_path
        else:
            source_path = benchmark_path
        source_manifest_path = legacy_manifest_path(source_path)
        manifest_path = benchmark_manifest_path(benchmark_path)
        logger.info("Migrating %s", source_path.name)

        benchmark = convert_benchmark(load_json_gz(source_path))
        route_count = sum(len(target.acceptable_routes) for target in benchmark.targets.values())
        logger.info("  targets=%s acceptable_routes=%s", len(benchmark.targets), route_count)

        if not args.write:
            continue
        source_for_manifest = source_manifest_path if source_manifest_path.exists() else source_path
        legacy_dir.mkdir(parents=True, exist_ok=True)
        if source_path == benchmark_path:
            move_to_legacy(benchmark_path, legacy_dir / benchmark_path.name, force=args.force)
            if manifest_path.exists():
                move_to_legacy(manifest_path, legacy_dir / manifest_path.name, force=args.force)
                source_for_manifest = legacy_dir / manifest_path.name
            else:
                source_for_manifest = legacy_dir / benchmark_path.name
        elif source_path == sibling_legacy_path:
            move_to_legacy(sibling_legacy_path, legacy_dir / benchmark_path.name, force=args.force)
            source_for_manifest = legacy_dir / benchmark_path.name
        save_benchmark(benchmark, benchmark_path)
        save_manifest(
            benchmark=benchmark,
            benchmark_path=benchmark_path,
            manifest_path=manifest_path,
            source_path=source_for_manifest,
            route_count=route_count,
            root_dir=definitions_dir.parents[1],
        )
        logger.info("  wrote %s and refreshed %s", benchmark_path.name, manifest_path.name)

    if not args.write:
        logger.info("Dry run complete. Re-run with --write to move v0.5.0 artifacts and write schema v2 files.")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=project_root / "data" / "retrocast" / "1-benchmarks" / "definitions",
    )
    parser.add_argument("--write", action="store_true", help="Move originals and write schema v2 benchmarks.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing v0.5.0 legacy artifacts.")
    return parser.parse_args()


def benchmark_manifest_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".json.gz") + ".manifest.json")


def legacy_sibling_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".json.gz") + ".rc0.6.json.gz")


def legacy_manifest_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".rc0.6.json.gz").removesuffix(".json.gz") + ".manifest.json")


def move_to_legacy(source: Path, destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"Legacy artifact already exists: {destination}")
        destination.unlink()
    source.rename(destination)


def save_manifest(
    *,
    benchmark: Benchmark,
    benchmark_path: Path,
    manifest_path: Path,
    source_path: Path,
    route_count: int,
    root_dir: Path,
) -> None:
    manifest = Manifest(
        action="scripts/migrations/03-migrate-benchmarks-to-schema-v2",
        parameters={"migration": "v0.5.0-benchmark-to-schema-v2", "legacy_dir": LEGACY_DIRNAME},
        source_files=[file_info(source_path, root_dir=root_dir)],
        output_files=[
            file_info(
                benchmark_path,
                root_dir=root_dir,
                content_hash=benchmark_content_hash(benchmark),
            )
        ],
        statistics={"n_targets": len(benchmark.targets), "n_acceptable_routes": route_count},
        summary={"schema_version": "2", "legacy_schema": "v0.5.0"},
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def file_info(path: Path, *, root_dir: Path, content_hash: str | None = None) -> FileInfo:
    return FileInfo(
        path=relative_path(path, root_dir=root_dir),
        file_hash=calculate_file_hash(path),
        content_hash=content_hash,
    )


def relative_path(path: Path, *, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def benchmark_content_hash(benchmark: Benchmark) -> str:
    payload = benchmark.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def convert_benchmark(data: dict[str, Any]) -> Benchmark:
    targets = {target_id: convert_target(target_id, target_data) for target_id, target_data in data["targets"].items()}
    return Benchmark(
        name=data["name"],
        description=data.get("description", ""),
        targets=targets,
        default_constraints=TaskConstraints(stock=data.get("stock_name")),
        annotations={"migrated_from": "v0.5.0"},
    )


def convert_target(target_id: str, data: dict[str, Any]) -> Target:
    return Target(
        id=target_id,
        smiles=SmilesStr(data["smiles"]),
        inchikey=InChIKeyStr(data["inchi_key"]),
        acceptable_routes=[convert_route(route) for route in data.get("acceptable_routes", [])],
        annotations=dict(data.get("metadata") or {}),
    )


def convert_route(data: dict[str, Any]) -> Route:
    annotations = dict(data.get("metadata") or {})
    rc06_fields = {
        key: data[key]
        for key in ("rank", "retrocast_version", "length", "has_convergent_reaction", "content_hash", "signature")
        if key in data
    }
    if rc06_fields:
        annotations["rc0.6"] = rc06_fields
    return Route(target=convert_molecule(data["target"]), annotations=annotations)


def convert_molecule(data: dict[str, Any]) -> Molecule:
    synthesis_step = data.get("synthesis_step")
    return Molecule(
        smiles=SmilesStr(data["smiles"]),
        inchikey=InChIKeyStr(data.get("inchikey") or data["inchi_key"]),
        product_of=convert_reaction(synthesis_step) if synthesis_step is not None else None,
        annotations=dict(data.get("metadata") or {}),
    )


def convert_reaction(data: dict[str, Any]) -> Reaction:
    return Reaction(
        reactants=[convert_molecule(reactant) for reactant in data["reactants"]],
        mapped_reaction_smiles=ReactionSmilesStr(data["mapped_smiles"]) if data.get("mapped_smiles") else None,
        template=data.get("template"),
        reagents=data.get("reagents"),
        solvents=data.get("solvents"),
        annotations=dict(data.get("metadata") or {}),
    )


if __name__ == "__main__":
    main()
