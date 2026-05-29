"""Migrate v0.5.0 benchmark definitions to schema v2.

Before writing, move the old benchmark artifacts out of the output directory:

    mkdir -p data/retrocast/1-benchmarks/definitions/v0.5.0
    mv data/retrocast/1-benchmarks/definitions/*.json.gz data/retrocast/1-benchmarks/definitions/v0.5.0/
    mv data/retrocast/1-benchmarks/definitions/*.manifest.json data/retrocast/1-benchmarks/definitions/v0.5.0/

By default this script only validates conversion. Pass ``--write`` to write the
schema v2 benchmarks and refreshed manifests at the original paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from retrocast.io import load_json_gz, save_benchmark
from retrocast.io.provenance import calculate_file_hash
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.models.route import Molecule, Reaction, Route
from retrocast.models.task import Benchmark, Target, TaskConstraints
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr
from retrocast.utils.logging import configure_script_logging, logger

LEGACY_DIRNAME = "v0.5.0"


def main() -> None:
    configure_script_logging()
    args = parse_args()
    definitions_dir = args.definitions_dir.resolve()
    legacy_dir = definitions_dir / LEGACY_DIRNAME
    root_dir = definitions_dir.parents[1]

    source_dir = legacy_dir
    benchmark_files = sorted(source_dir.glob("*.json.gz"))
    if not benchmark_files and args.write:
        raise FileNotFoundError(f"Move legacy benchmark artifacts into {legacy_dir} before running with --write.")
    if not benchmark_files:
        source_dir = definitions_dir
        benchmark_files = sorted(source_dir.glob("*.json.gz"))
    logger.info("Found %s benchmark definitions in %s", len(benchmark_files), source_dir)

    for source_path in benchmark_files:
        output_path = definitions_dir / source_path.name
        manifest_path = definitions_dir / source_path.name.replace(".json.gz", ".manifest.json")
        logger.info("Migrating %s", source_path.name)

        benchmark = convert_benchmark(load_json_gz(source_path))
        route_count = sum(len(target.acceptable_routes) for target in benchmark.targets.values())
        logger.info("  targets=%s acceptable_routes=%s", len(benchmark.targets), route_count)

        if not args.write:
            continue
        save_benchmark(benchmark, output_path)
        manifest = Manifest(
            action="scripts/migrations/03-migrate-benchmarks-to-schema-v2",
            parameters={"migration": "v0.5.0-benchmark-to-schema-v2", "legacy_dir": LEGACY_DIRNAME},
            source_files=[
                FileInfo(
                    path=str(source_path.relative_to(root_dir)),
                    file_hash=calculate_file_hash(source_path),
                )
            ],
            output_files=[
                FileInfo(
                    path=str(output_path.relative_to(root_dir)),
                    file_hash=calculate_file_hash(output_path),
                    content_hash=benchmark_content_hash(benchmark),
                )
            ],
            statistics={"n_targets": len(benchmark.targets), "n_acceptable_routes": route_count},
            summary={"schema_version": "2", "legacy_schema": "v0.5.0"},
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info("  wrote %s and refreshed %s", output_path.name, manifest_path.name)

    if not args.write:
        logger.info("Dry run complete. Move legacy artifacts as documented, then re-run with --write.")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=project_root / "data" / "retrocast" / "1-benchmarks" / "definitions",
    )
    parser.add_argument("--write", action="store_true", help="Write schema v2 benchmarks and manifests.")
    return parser.parse_args()


def benchmark_content_hash(benchmark: Benchmark) -> str:
    payload = benchmark.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def convert_benchmark(data: dict[str, Any]) -> Benchmark:
    targets = {target_id: convert_target(target_id, target_data) for target_id, target_data in data["targets"].items()}
    return Benchmark(
        name=data["name"],
        description=data.get("description") or "",
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
