import gzip
import json
import logging
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
        if not path.exists():
            raise ArtifactNotFoundError(
                f"File not found: {path}",
                code="io.not_found",
                context={"path": str(path)},
            )
        try:
            rows: list[Any] = []
            with open(path, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line_text = line.strip()
                    if not line_text:
                        continue
                    try:
                        rows.append(json.loads(line_text))
                    except json.JSONDecodeError as e:
                        raise ArtifactDecodeError(
                            f"Failed to decode JSONL row {line_number} from {path}: {e}",
                            code="io.decode_failed",
                            context={"path": str(path), "line_number": line_number, "line_text": line_text},
                        ) from e
            return rows
        except OSError as e:
            raise ArtifactDecodeError(
                f"Failed to load {path}: {e}",
                code="io.decode_failed",
                context={"path": str(path)},
            ) from e

    if name.endswith((".json", ".json.gz")):
        if path.suffix == ".gz":
            return load_json_gz(path)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"File not found: {path}",
                code="io.not_found",
                context={"path": str(path)},
            )
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as e:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, BaseModel):
            json_obj = data.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
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
                    json_obj = row.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
                else:
                    json_obj = row
                payload = json.dumps(json_obj, sort_keys=True, separators=(",", ":"))
                gz_f.write(payload.encode("utf-8"))
                gz_f.write(b"\n")
                n_rows += 1
        logger.debug(f"Saved {n_rows} rows to {path}")
        return n_rows
    except (OSError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


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
    except (OSError, AttributeError, TypeError, ValueError) as e:
        raise ArtifactWriteError(
            f"Failed to save {path}: {e}",
            code="io.write_failed",
            context={"path": str(path)},
        ) from e


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


def load_jsonl_gz(path: Path, skip_empty: bool = True) -> list[Any]:
    """Loads JSON rows from a gzipped JSONL file, optionally rejecting empty lines."""
    return list(iter_jsonl_gz(path, skip_empty=skip_empty))


def iter_jsonl_gz(path: Path, skip_empty: bool = True) -> Iterator[Any]:
    """Streams JSON rows from a gzipped JSONL file."""
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"File not found: {path}",
            code="io.not_found",
            context={"path": str(path)},
        )

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    if not skip_empty:
                        raise ArtifactDecodeError(
                            f"Empty JSONL row {line_number} in {path}",
                            code="io.decode_failed",
                            context={"path": str(path), "line_number": line_number},
                        )
                    continue
                try:
                    yield json.loads(line)
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


def load_lines_gz(path: Path) -> list[str]:
    """Loads newline-delimited text lines from a gzip file."""
    return list(iter_lines_gz(path))


def iter_lines_gz(path: Path) -> Iterator[str]:
    """Streams newline-delimited text lines from a gzip file."""
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"File not found: {path}",
            code="io.not_found",
            context={"path": str(path)},
        )

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
    except OSError as e:
        raise ArtifactDecodeError(
            f"Failed to load {path}: {e}",
            code="io.decode_failed",
            context={"path": str(path)},
        ) from e
