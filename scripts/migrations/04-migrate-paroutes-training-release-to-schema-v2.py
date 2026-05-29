"""Migrate the v2026-05-12 PaRoutes training release artifacts to schema v2.

This modifies the release in place when passed ``--write``:

    uv run scripts/migrations/04-migrate-paroutes-training-release-to-schema-v2.py --write

The migration is intentionally narrow: it only updates the checked-in
``v2026-05-12`` training-release wire format so runtime training models do not
need legacy loaders.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from retrocast.curation.training.reaction_release import reaction_records_content_hash
from retrocast.curation.training.records import TrainingReactionRecord, TrainingRouteRecord
from retrocast.curation.training.route_release import route_records_content_hash
from retrocast.io import iter_jsonl_gz, save_jsonl_gz
from retrocast.io.provenance import calculate_file_hash
from retrocast.utils.logging import configure_script_logging, logger

ROUTE_RELEASES = ("route-holdout-n1-n5", "reaction-holdout-n1-n5")
ROUTE_FILES = ("all.jsonl.gz", "training.jsonl.gz", "validation.jsonl.gz")
SINGLE_STEP_RELEASE = "single-step-reaction-holdout-n1-n5"
SINGLE_STEP_JSONL_FILES = ("all.jsonl.gz", "training.jsonl.gz", "validation.jsonl.gz")
MIGRATION_NAME = "scripts/migrations/04-migrate-paroutes-training-release-to-schema-v2"


def main() -> None:
    configure_script_logging()
    args = parse_args()
    release_root = args.release_root.resolve()
    project_root = Path(__file__).resolve().parents[2]

    reaction_ids_by_route = build_reaction_id_index(release_root / "reaction-holdout-n1-n5" / "all.jsonl.gz")

    for release_name in ROUTE_RELEASES:
        release_dir = release_root / release_name
        for filename in ROUTE_FILES:
            migrate_jsonl_file(
                release_dir / filename,
                convert=lambda row: convert_route_record(row),
                write=args.write,
            )

    single_step_dir = release_root / SINGLE_STEP_RELEASE
    for filename in SINGLE_STEP_JSONL_FILES:
        migrate_jsonl_file(
            single_step_dir / filename,
            convert=lambda row: convert_reaction_record(row, reaction_ids_by_route),
            write=args.write,
        )

    if args.write:
        for release_name in ROUTE_RELEASES:
            refresh_manifest(release_root / release_name / "manifest.json", project_root=project_root)
        refresh_manifest(single_step_dir / "manifest.json", project_root=project_root)
        logger.info("Migrated %s in place", release_root)
    else:
        logger.info("Dry run complete. Re-run with --write to modify %s in place.", release_root)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=project_root / "data" / "retrocast" / "releases" / "paroutes-training-sets" / "v2026-05-12",
    )
    parser.add_argument("--write", action="store_true", help="Rewrite release artifacts in place.")
    return parser.parse_args()


def migrate_jsonl_file(path: Path, *, convert, write: bool) -> None:
    rows = [convert(row) for row in iter_jsonl_gz(path)]
    logger.info("%s: %s records", display_path(path), f"{len(rows):,}")
    if not write:
        return
    tmp_path = path.with_name(f"{path.name}.tmp")
    save_jsonl_gz(rows, tmp_path)
    tmp_path.replace(path)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def convert_route_record(row: Mapping[str, Any]) -> TrainingRouteRecord:
    payload = {
        "id": row["id"],
        "split": row["split"],
        "route": convert_route(row["route"]),
        "sources": convert_sources(row),
    }
    return TrainingRouteRecord.model_validate(payload)


def convert_sources(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("sources"), list):
        return list(row["sources"])
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"route record {row.get('id')} has neither sources nor legacy source")

    raw_indices = require_sequence(source, "raw_indices", row.get("id"))
    raw_hashes = require_sequence(source, "raw_route_hashes", row.get("id"))
    patent_ids = require_sequence(source, "patent_ids", row.get("id"))
    if len(raw_indices) != len(raw_hashes) or len(raw_indices) != len(patent_ids):
        raise ValueError(f"route record {row.get('id')} has mismatched legacy source arrays")

    dataset = source.get("dataset") or "unknown"
    sources = []
    for raw_index, raw_hash, patent_id in zip(raw_indices, raw_hashes, patent_ids, strict=True):
        item = {"dataset": dataset, "raw_index": raw_index, "raw_route_hash": raw_hash}
        if patent_id is not None:
            item["patent_id"] = patent_id
        sources.append(item)
    return sources


def require_sequence(source: Mapping[str, Any], key: str, record_id: object) -> list[Any]:
    value = source.get(key)
    if not isinstance(value, list):
        raise ValueError(f"route record {record_id} legacy source.{key} must be a list")
    return value


def convert_route(route: Mapping[str, Any]) -> dict[str, Any]:
    if route.get("schema_version") == "2":
        return dict(route)
    return {
        "target": convert_molecule(route["target"]),
        "annotations": dict(route.get("metadata") or {}),
        "schema_version": "2",
    }


def convert_molecule(molecule: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "smiles": molecule["smiles"],
        "inchikey": molecule.get("inchikey") or molecule.get("inchi_key"),
        "annotations": dict(molecule.get("metadata") or molecule.get("annotations") or {}),
    }
    reaction = molecule.get("product_of") or molecule.get("synthesis_step")
    if reaction is not None:
        payload["product_of"] = convert_reaction(reaction)
    return payload


def convert_reaction(reaction: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "reactants": [convert_molecule(reactant) for reactant in reaction["reactants"]],
        "annotations": dict(reaction.get("metadata") or reaction.get("annotations") or {}),
    }
    mapped_smiles = reaction.get("mapped_reaction_smiles") or reaction.get("mapped_smiles")
    if mapped_smiles is not None:
        payload["mapped_reaction_smiles"] = mapped_smiles
    for key in ("template", "reagents", "solvents"):
        if reaction.get(key) is not None:
            payload[key] = reaction[key]
    return payload


def build_reaction_id_index(route_all_path: Path) -> dict[str, list[str]]:
    return {row["id"]: list(iter_reaction_ids(row["route"]["target"])) for row in iter_jsonl_gz(route_all_path)}


def iter_reaction_ids(molecule: Mapping[str, Any], indices: tuple[int, ...] = ()) -> Iterable[str]:
    reaction = molecule.get("product_of") or molecule.get("synthesis_step")
    if reaction is None:
        return
    suffix = "/".join(str(index) for index in indices)
    yield f"rc:r:/{suffix}" if suffix else "rc:r:/"
    for index, reactant in enumerate(reaction["reactants"]):
        yield from iter_reaction_ids(reactant, (*indices, index))


def convert_reaction_record(
    row: Mapping[str, Any], reaction_ids_by_route: Mapping[str, list[str]]
) -> TrainingReactionRecord:
    payload = dict(row)
    payload["sources"] = [convert_reaction_source(source, reaction_ids_by_route, row["id"]) for source in row["sources"]]
    return TrainingReactionRecord.model_validate(payload)


def convert_reaction_source(
    source: Mapping[str, Any], reaction_ids_by_route: Mapping[str, list[str]], record_id: str
) -> dict[str, Any]:
    payload = dict(source)
    if payload.get("reaction_id") is not None:
        return payload

    route_id = payload["route_id"]
    step_index = payload.get("step_index")
    if not isinstance(step_index, int) or step_index < 1:
        raise ValueError(f"reaction record {record_id} source for {route_id} has invalid step_index {step_index!r}")
    reaction_ids = reaction_ids_by_route.get(route_id)
    if reaction_ids is None or step_index > len(reaction_ids):
        raise ValueError(f"reaction record {record_id} source references missing route step {route_id}:{step_index}")
    payload["reaction_id"] = reaction_ids[step_index - 1]
    return payload


def refresh_manifest(manifest_path: Path, *, project_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2"
    migrations = manifest.setdefault("summary", {}).setdefault("schema_migrations", [])
    if MIGRATION_NAME not in migrations:
        migrations.append(MIGRATION_NAME)
    refresh_file_infos(manifest.get("source_files", []), project_root=project_root)
    output_files = manifest.get("output_files", {})
    if isinstance(output_files, dict):
        refresh_file_infos(output_files.values(), project_root=project_root)
        if manifest_path.parent.name in ROUTE_RELEASES:
            refresh_route_content_hashes(output_files, project_root=project_root)
        if manifest_path.parent.name == SINGLE_STEP_RELEASE:
            refresh_reaction_content_hashes(output_files, project_root=project_root)
    elif isinstance(output_files, list):
        refresh_file_infos(output_files, project_root=project_root)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_file_infos(file_infos: Iterable[dict[str, Any]], *, project_root: Path) -> None:
    for file_info in file_infos:
        if "sha256" in file_info:
            file_info.pop("sha256")
        path = project_root / file_info["path"]
        file_info["file_hash"] = calculate_file_hash(path)
        file_info.setdefault("content_hash", None)


def refresh_route_content_hashes(output_files: Mapping[str, dict[str, Any]], *, project_root: Path) -> None:
    for label in ("all", "training", "validation"):
        file_info = output_files.get(label)
        if file_info is None:
            continue
        records = [TrainingRouteRecord.model_validate(row) for row in iter_jsonl_gz(project_root / file_info["path"])]
        file_info["content_hash"] = route_records_content_hash(records)


def refresh_reaction_content_hashes(output_files: Mapping[str, dict[str, Any]], *, project_root: Path) -> None:
    for label in ("all", "training", "validation"):
        file_info = output_files.get(label)
        if file_info is None:
            continue
        records = [TrainingReactionRecord.model_validate(row) for row in iter_jsonl_gz(project_root / file_info["path"])]
        file_info["content_hash"] = reaction_records_content_hash(records)


if __name__ == "__main__":
    main()
