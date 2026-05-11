import gzip
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from retrocast.exceptions import ArtifactDecodeError, ArtifactNotFoundError, ArtifactWriteError

logger = logging.getLogger(__name__)


def save_json_gz(data: Any, path: Path) -> None:
    """
    Serializes data to a gzipped JSON file.
    If data is a Pydantic model, dumps it first.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, BaseModel):
            json_obj = data.model_dump(mode="json")
        else:
            json_obj = data
        json_str = json.dumps(json_obj, indent=2)
        with open(path, "wb") as raw_f, gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
            gz_f.write(json_str.encode("utf-8"))
        logger.debug(f"Saved {path}")
    except (OSError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


def save_jsonl_gz(rows: Iterable[Any], path: Path) -> int:
    """Serializes JSON rows to a deterministic gzipped JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    try:
        with open(path, "wb") as raw_f, gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
            for row in rows:
                if isinstance(row, BaseModel):
                    json_obj = row.model_dump(mode="json")
                else:
                    json_obj = row
                payload = json.dumps(json_obj, sort_keys=True, separators=(",", ":"))
                gz_f.write(payload.encode("utf-8"))
                gz_f.write(b"\n")
                n_rows += 1
        logger.debug(f"Saved {n_rows} rows to {path}")
        return n_rows
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        raise


def save_lines_gz(lines: Iterable[str], path: Path) -> int:
    """Writes newline-delimited text lines to a deterministic gzip file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    try:
        with open(path, "wb") as raw_f, gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
            for line in lines:
                gz_f.write(line.encode("utf-8"))
                gz_f.write(b"\n")
                n_lines += 1
        logger.debug(f"Saved {n_lines} lines to {path}")
        return n_lines
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        raise


def load_json_gz(path: Path) -> Any:
    """Loads a gzipped JSON file."""
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"File not found: {path}",
            code="io.not_found",
            context={"path": str(path)},
        )

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e


def load_jsonl_gz(path: Path) -> list[Any]:
    """Loads JSON rows from a gzipped JSONL file."""
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"File not found: {path}",
            code="io.not_found",
            context={"path": str(path)},
        )

    rows: list[Any] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ArtifactDecodeError(
                        f"Failed to decode JSONL row {line_number} from {path}: {e}",
                        code="io.decode_failed",
                        context={"path": str(path), "line_number": line_number},
                    ) from e
    except OSError as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e

    return rows


def load_lines_gz(path: Path) -> list[str]:
    """Loads newline-delimited text lines from a gzip file."""
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"File not found: {path}",
            code="io.not_found",
            context={"path": str(path)},
        )

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    except OSError as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e
