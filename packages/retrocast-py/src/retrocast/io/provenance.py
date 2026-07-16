from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from retrocast.hashing import hash_file
from retrocast.models.provenance import Manifest

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
LabeledManifestOutputWithHash: TypeAlias = tuple[str, Path, Any, ContentType | ContentTypeHint, str]
ManifestOutput: TypeAlias = UnlabeledManifestOutput | LabeledManifestOutput | LabeledManifestOutputWithHash


def calculate_file_hash(path: Path) -> str:
    try:
        return hash_file(path)
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
    from retrocast import native

    for path in sources:
        if not path.exists():
            raise FileNotFoundError(f"manifest source file not found: {path}")
        _relative_path(path, root_dir)

    normalized_outputs = []
    for output in outputs:
        label, path, obj, content_type, content_hash = _normalize_manifest_output(output)
        content_type = ContentType(content_type)
        _relative_path(path, root_dir)
        normalized_outputs.append(
            {
                "label": label,
                "path": str(path),
                "value": _jsonable(obj),
                "content_type": content_type.value,
                "content_hash": content_hash,
            }
        )
    if keyed_output_files:
        for output in normalized_outputs:
            if output["label"] is None:
                raise ValueError("keyed_output_files=True requires every output to include a label")
    return native.create_manifest(
        {
            "action": action,
            "sources": [str(path) for path in sources],
            "outputs": normalized_outputs,
            "root_dir": str(root_dir),
            "parameters": parameters or {},
            "statistics": statistics or {},
            "directives": directives or {},
            "summary": summary or {},
            "release_name": release_name,
            "keyed_output_files": keyed_output_files,
        }
    )


def _normalize_manifest_output(
    output: ManifestOutput,
) -> tuple[str | None, Path, Any, ContentType | ContentTypeHint, str | None]:
    items: tuple[Any, ...] = tuple(output)
    if len(items) == 3:
        path, obj, content_type = items
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        return None, path, obj, content_type, None
    if len(items) == 4:
        label, path, obj, content_type = items
        if not isinstance(label, str):
            raise TypeError(f"manifest output label must be str, got {type(label).__name__}")
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        return label, path, obj, content_type, None
    if len(items) == 5:
        label, path, obj, content_type, content_hash = items
        if not isinstance(label, str):
            raise TypeError(f"manifest output label must be str, got {type(label).__name__}")
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        if not isinstance(content_hash, str):
            raise TypeError(f"manifest output content hash must be str, got {type(content_hash).__name__}")
        return label, path, obj, content_type, content_hash
    raise ValueError(f"manifest output must have 3, 4, or 5 items, got {len(items)}")


def _relative_path(path: Path, root_dir: Path) -> str:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        logger.warning("path %s is not inside root %s; storing absolute path", resolved_path, resolved_root)
        return str(resolved_path)


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj
