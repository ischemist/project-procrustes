from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from retrocast._version import __version__
from retrocast.models.provenance import FileInfo, Manifest

logger = logging.getLogger(__name__)


class ContentType(StrEnum):
    BENCHMARK = "benchmark"
    PREDICTIONS = "predictions"
    ROUTE_CORPUS = "route_corpus"
    STOCK = "stock"
    UNKNOWN = "unknown"


ContentTypeHint = Literal["benchmark", "predictions", "route_corpus", "stock", "unknown"]
UnlabeledManifestOutput: TypeAlias = tuple[Path, Any, ContentType | ContentTypeHint]
LabeledManifestOutput: TypeAlias = tuple[str, Path, Any, ContentType | ContentTypeHint]
ManifestOutput: TypeAlias = UnlabeledManifestOutput | LabeledManifestOutput


def calculate_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError:
        logger.exception("could not hash file %s", path)
        return "error-hashing-file"


def create_manifest(
    action: str,
    sources: list[Path],
    outputs: list[ManifestOutput],
    root_dir: Path,
    parameters: dict[str, Any] | None = None,
    statistics: dict[str, Any] | None = None,
    directives: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    release_name: str | None = None,
    keyed_output_files: bool = False,
) -> Manifest:
    source_infos = [_required_file_info(path, root_dir) for path in sources]

    output_infos = []
    for output in outputs:
        label, path, obj, content_type = _normalize_manifest_output(output)
        file_hash = calculate_file_hash(path) if path.exists() else "file-not-written"
        output_infos.append(
            FileInfo(
                label=label,
                path=_relative_path(path, root_dir),
                file_hash=file_hash,
                content_hash=_content_hash(obj, ContentType(content_type)),
            )
        )

    manifest_outputs: list[FileInfo] | dict[str, FileInfo]
    if keyed_output_files:
        manifest_outputs = {}
        for file_info in output_infos:
            if file_info.label is None:
                raise ValueError("keyed_output_files=True requires every output to include a label")
            manifest_outputs[file_info.label] = file_info
    else:
        manifest_outputs = output_infos

    return Manifest(
        schema_version="2",
        retrocast_version=__version__,
        created_at=datetime.now(UTC),
        action=action,
        parameters=parameters or {},
        directives=directives or {},
        release_name=release_name,
        source_files=source_infos,
        output_files=manifest_outputs,
        statistics=statistics or {},
        summary=summary or {},
    )


def _normalize_manifest_output(
    output: ManifestOutput,
) -> tuple[str | None, Path, Any, ContentType | ContentTypeHint]:
    items: tuple[Any, ...] = tuple(output)
    if len(items) == 3:
        path, obj, content_type = items
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        return None, path, obj, content_type
    if len(items) == 4:
        label, path, obj, content_type = items
        if not isinstance(label, str):
            raise TypeError(f"manifest output label must be str, got {type(label).__name__}")
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        return label, path, obj, content_type
    raise ValueError(f"manifest output must have 3 or 4 items, got {len(items)}")


def _relative_path(path: Path, root_dir: Path) -> str:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        logger.warning("path %s is not inside root %s; storing absolute path", resolved_path, resolved_root)
        return str(resolved_path)


def _required_file_info(path: Path, root_dir: Path) -> FileInfo:
    if not path.exists():
        raise FileNotFoundError(f"manifest source file not found: {path}")
    return FileInfo(path=_relative_path(path, root_dir), file_hash=calculate_file_hash(path))


def _content_hash(obj: Any, content_type: ContentType) -> str | None:
    if content_type == ContentType.UNKNOWN:
        return None
    if content_type == ContentType.STOCK and isinstance(obj, dict):
        payload = sorted(obj.keys())
    else:
        payload = _jsonable(obj)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj
