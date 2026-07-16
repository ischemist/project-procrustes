from __future__ import annotations

from pathlib import Path
from typing import Any

from retrocast.io.provenance import ContentType, create_manifest


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
    manifest = create_manifest(
        action=action,
        sources=sources,
        outputs=[(output, None, ContentType.UNKNOWN) for output in outputs],
        root_dir=root_dir,
        parameters=parameters,
        statistics=statistics,
        summary=summary,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def manifest_sidecar_path(output_path: Path) -> Path:
    for suffix in (".jsonl.gz", ".json.gz", ".csv.gz", ".txt.gz"):
        if output_path.name.endswith(suffix):
            return output_path.with_name(f"{output_path.name.removesuffix(suffix)}.manifest.json")
    return output_path.with_name(f"{output_path.stem}.manifest.json")
