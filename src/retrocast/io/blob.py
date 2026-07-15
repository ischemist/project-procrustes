import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from retrocast.exceptions import ArtifactDecodeError, ArtifactNotFoundError, ArtifactWriteError

logger = logging.getLogger(__name__)


def load_json_artifact(path: Path) -> Any:
    """Load `.json`, `.json.gz`, `.jsonl`, or `.jsonl.gz` artifacts."""
    path = Path(path)
    name = path.name

    if name.endswith((".jsonl", ".jsonl.gz")):
        if path.suffix == ".gz":
            return load_jsonl_gz(path)
        return list(_iter_jsonl(path, compressed=False))

    if name.endswith((".json", ".json.gz")):
        _ensure_exists(path)
        try:
            from retrocast import native

            return native.read_json(str(path))
        except (OSError, RuntimeError, json.JSONDecodeError) as e:
            raise ArtifactDecodeError(
                f"Failed to load {path}: {e}",
                code="io.decode_failed",
                context={"path": str(path)},
            ) from e

    raise ArtifactDecodeError(
        f"Unsupported JSON artifact format: {path}",
        code="io.decode_failed",
        context={"path": str(path)},
    )


def save_json_gz(data: Any, path: Path) -> None:
    """
    Serializes data to a gzipped JSON file.
    If data is a Pydantic model, dumps it first.
    """
    path = Path(path)
    try:
        if isinstance(data, BaseModel):
            json_obj = data.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
        else:
            json_obj = data
        from retrocast import native

        native.write_json_gz(str(path), json_obj)
        logger.debug(f"Saved {path}")
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


def save_jsonl_gz(rows: Iterable[Any], path: Path) -> int:
    """Serializes JSON rows to a deterministic gzipped JSONL file."""
    path = Path(path)

    n_rows = 0
    try:
        serialized = [
            row.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
            if isinstance(row, BaseModel)
            else row
            for row in rows
        ]
        from retrocast import native

        n_rows = native.write_jsonl_gz(str(path), serialized)
        logger.debug(f"Saved {n_rows} rows to {path}")
        return n_rows
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


def save_lines_gz(lines: Iterable[str], path: Path) -> int:
    """Writes newline-delimited text lines to a deterministic gzip file."""
    path = Path(path)

    n_lines = 0
    try:
        from retrocast import native

        n_lines = native.write_lines_gz(str(path), list(lines))
        logger.debug(f"Saved {n_lines} lines to {path}")
        return n_lines
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


def save_csv_gz(rows: Iterable[Iterable[Any]], path: Path) -> int:
    """Writes CSV rows to a deterministic gzip file."""
    path = Path(path)

    n_rows = 0
    try:
        from retrocast import native

        serialized = [["" if value is None else str(value) for value in row] for row in rows]
        n_rows = native.write_csv_gz(str(path), serialized)
        logger.debug(f"Saved {n_rows} CSV rows to {path}")
        return n_rows
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


def load_json_gz(path: Path) -> Any:
    """Loads a gzipped JSON file."""
    path = Path(path)
    _ensure_exists(path)

    try:
        from retrocast import native

        return native.read_json(str(path))
    except (EOFError, OSError, RuntimeError, json.JSONDecodeError) as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e


def load_jsonl_gz(path: Path, skip_empty: bool = True) -> list[Any]:
    """Loads JSON rows from a gzipped JSONL file, optionally rejecting empty lines."""
    return list(iter_jsonl_gz(path, skip_empty=skip_empty))


def iter_jsonl_gz(path: Path, skip_empty: bool = True) -> Iterator[Any]:
    """Streams JSON rows from a gzipped JSONL file."""
    yield from _iter_jsonl(path, compressed=True, skip_empty=skip_empty)


def _iter_jsonl(path: Path, *, compressed: bool, skip_empty: bool = True) -> Iterator[Any]:
    """Streams JSONL rows from plain text or gzip while preserving row context."""
    path = Path(path)
    _ensure_exists(path)

    try:
        from retrocast import native

        yield from native.read_jsonl(str(path), skip_empty=skip_empty)
    except RuntimeError as e:
        match = re.search(r"JSONL row (\d+):", str(e))
        line_number = int(match.group(1)) if match else None
        context: dict[str, Any] = {"path": str(path)}
        if line_number is not None:
            context["line_number"] = line_number
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context=context,
        ) from e
    except (EOFError, OSError) as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e


def load_lines_gz(path: Path) -> list[str]:
    """Loads newline-delimited text lines from a gzip file."""
    return list(iter_lines_gz(path))


def iter_lines_gz(path: Path) -> Iterator[str]:
    """Streams newline-delimited text lines from a gzip file."""
    path = Path(path)
    _ensure_exists(path)

    try:
        from retrocast import native

        yield from native.read_lines_gz(str(path))
    except (EOFError, OSError, RuntimeError) as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e


def _ensure_exists(path: Path) -> None:
    if path.exists():
        return
    raise ArtifactNotFoundError(
        f"File not found: {path}",
        code="io.not_found",
        context={"path": str(path)},
    )
