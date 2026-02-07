"""Dynamic adapter resolution via self-describing data (manifest directives).

Resolution hierarchy (strict priority):
1. CLI override (--adapter flag)
2. Manifest directives (manifest.json -> directives.adapter)
3. Failure (AdapterResolutionError — no guessing, no heuristics)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.exceptions import AdapterResolutionError
from retrocast.paths import validate_filename

logger = logging.getLogger(__name__)

DEFAULT_RAW_RESULTS_FILENAME = "results.json.gz"


def _read_manifest_directives(raw_dir: Path) -> dict[str, Any]:
    """Read directives from a manifest.json in the given directory.

    Returns an empty dict if the file is missing, malformed, or has no directives.
    Never raises — all errors are logged and swallowed.
    """
    manifest_path = raw_dir / "manifest.json"

    if not manifest_path.exists():
        logger.debug(f"No manifest.json found in {raw_dir}")
        return {}

    try:
        with open(manifest_path) as f:
            manifest_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read manifest at {manifest_path}: {e}")
        return {}

    directives = manifest_data.get("directives", {})
    if not isinstance(directives, dict):
        logger.warning(f"Manifest directives is not a dict in {manifest_path}")
        return {}

    return directives


def resolve_adapter(
    *,
    cli_adapter: str | None,
    raw_dir: Path,
    model_name: str,
) -> tuple[BaseAdapter, str]:
    """Resolve an adapter instance using the strict priority hierarchy.

    Args:
        cli_adapter: Adapter name from --adapter CLI flag, or None.
        raw_dir: Path to the raw data directory (e.g., data/2-raw/{model}/{benchmark}/).
        model_name: Model name (for error messages only).

    Returns:
        Tuple of (adapter_instance, resolution_source) where source is one of
        "cli" or "manifest".

    Raises:
        AdapterResolutionError: If no adapter can be resolved from any source.
    """
    # Avoid circular import — ADAPTER_MAP is built at import time in __init__
    from retrocast.adapters import ADAPTER_MAP, get_adapter

    # 1. CLI override (explicit intent)
    if cli_adapter is not None:
        if cli_adapter not in ADAPTER_MAP:
            raise AdapterResolutionError(
                f"CLI adapter '{cli_adapter}' is not a valid adapter. Available: {sorted(ADAPTER_MAP.keys())}"
            )
        logger.info(f"Resolved adapter '{cli_adapter}' from cli")
        return get_adapter(cli_adapter), "cli"

    # 2. Manifest directives (self-describing data)
    directives = _read_manifest_directives(raw_dir)
    manifest_adapter = directives.get("adapter")

    if manifest_adapter is not None:
        if manifest_adapter not in ADAPTER_MAP:
            raise AdapterResolutionError(
                f"Manifest in {raw_dir} declares adapter '{manifest_adapter}', "
                f"but it is not a valid adapter. Available: {sorted(ADAPTER_MAP.keys())}"
            )
        logger.info(f"Resolved adapter '{manifest_adapter}' from manifest")
        return get_adapter(manifest_adapter), "manifest"

    # 3. Failure — no guessing
    raise AdapterResolutionError(
        f"Cannot resolve adapter for model '{model_name}' in {raw_dir}. "
        f"No --adapter flag provided, and no manifest.json with directives.adapter found. "
        f"Either pass --adapter explicitly or ensure the raw data directory contains a "
        f'manifest.json with \'"directives": {{"adapter": "<name>"}}\'.'
    )


def resolve_raw_results_filename(*, raw_dir: Path) -> str:
    """Resolve the raw results filename from manifest directives.

    Falls back to the default 'results.json.gz' if not specified in the manifest.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        The filename string (e.g., "results.json.gz" or "valid_results.json.gz").

    Raises:
        SecurityError: If the filename contains path traversal sequences.
    """
    directives = _read_manifest_directives(raw_dir)
    filename = directives.get("raw_results_filename", DEFAULT_RAW_RESULTS_FILENAME)

    # Security: Validate filename to prevent path traversal attacks
    filename = validate_filename(filename, param_name="raw_results_filename")

    if filename != DEFAULT_RAW_RESULTS_FILENAME:
        logger.debug(f"Using raw_results_filename '{filename}' from manifest directives")
    return filename
