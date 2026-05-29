from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retrocast import __version__
from retrocast.io.provenance import calculate_file_hash
from retrocast.models.provenance import FileInfo, Manifest


def write_manifest(
    path: Path,
    *,
    action: str,
    sources: list[Path],
    outputs: list[Path],
    root_dir: Path,
    parameters: dict[str, Any] | None = None,
    statistics: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    manifest = Manifest(
        schema_version="2",
        retrocast_version=__version__,
        created_at=datetime.now(UTC),
        action=action,
        parameters=parameters or {},
        source_files=[_file_info(source, root_dir) for source in sources if source.exists()],
        output_files=[_file_info(output, root_dir) for output in outputs],
        statistics=statistics or {},
        summary=summary or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def manifest_sidecar_path(output_path: Path) -> Path:
    for suffix in (".jsonl.gz", ".json.gz", ".csv.gz", ".txt.gz"):
        if output_path.name.endswith(suffix):
            return output_path.with_name(f"{output_path.name.removesuffix(suffix)}.manifest.json")
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def _file_info(path: Path, root_dir: Path) -> FileInfo:
    return FileInfo(path=_relative_path(path, root_dir), file_hash=calculate_file_hash(path))


def _relative_path(path: Path, root_dir: Path) -> str:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)
