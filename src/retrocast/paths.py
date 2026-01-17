"""Centralized path resolution for RetroCast data directories."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path("data/retrocast")
LEGACY_DATA_DIR = Path("data")  # Old default for migration detection
ENV_VAR_NAME = "RETROCAST_DATA_DIR"


def resolve_data_dir(
    cli_arg: str | Path | None = None,
    config_value: str | Path | None = None,
) -> Path:
    """
    Resolve the data directory with priority:
    1. CLI argument (--data-dir)
    2. Environment variable (RETROCAST_DATA_DIR)
    3. Config file (data_dir key)
    4. Default (data/retrocast/)

    Args:
        cli_arg: Value from --data-dir CLI flag
        config_value: Value from config file's data_dir key

    Returns:
        Resolved Path to data directory
    """
    if cli_arg is not None:
        return Path(cli_arg)

    env_value = os.environ.get(ENV_VAR_NAME)
    if env_value:
        return Path(env_value)

    if config_value is not None:
        return Path(config_value)

    return DEFAULT_DATA_DIR


def get_paths(data_dir: Path) -> dict[str, Path]:
    """
    Return standard directory layout from a base data directory.

    Args:
        data_dir: Base data directory path

    Returns:
        Dictionary mapping logical names to resolved paths:
        - benchmarks: {data_dir}/1-benchmarks/definitions
        - stocks: {data_dir}/1-benchmarks/stocks
        - raw: {data_dir}/2-raw
        - processed: {data_dir}/3-processed
        - scored: {data_dir}/4-scored
        - results: {data_dir}/5-results
    """
    return {
        "benchmarks": data_dir / "1-benchmarks" / "definitions",
        "stocks": data_dir / "1-benchmarks" / "stocks",
        "raw": data_dir / "2-raw",
        "processed": data_dir / "3-processed",
        "scored": data_dir / "4-scored",
        "results": data_dir / "5-results",
    }


def check_migration_needed(resolved_dir: Path) -> str | None:
    """
    Check if data exists at legacy location but not at resolved location.

    This helps users who have existing data in the old default location (data/)
    understand that they need to migrate to the new default (data/retrocast/).

    Args:
        resolved_dir: The resolved data directory

    Returns:
        Warning message if migration is needed, None otherwise
    """
    # Only warn if using the new default
    if resolved_dir != DEFAULT_DATA_DIR:
        return None

    legacy_benchmarks = LEGACY_DATA_DIR / "1-benchmarks"
    new_benchmarks = resolved_dir / "1-benchmarks"

    if legacy_benchmarks.exists() and not new_benchmarks.exists():
        return (
            f"Found data at legacy location '{LEGACY_DATA_DIR}/' but not at "
            f"'{resolved_dir}/'. Consider moving your data:\n"
            f"  mv {LEGACY_DATA_DIR}/1-benchmarks {resolved_dir}/\n"
            f"  mv {LEGACY_DATA_DIR}/2-raw {resolved_dir}/\n"
            f"  ...\n"
            f"Or set {ENV_VAR_NAME}={LEGACY_DATA_DIR} to keep using the old location."
        )
    return None


def get_data_dir_source(
    cli_arg: str | Path | None = None,
    config_value: str | Path | None = None,
) -> str:
    """
    Return a human-readable description of where the data directory came from.

    Args:
        cli_arg: Value from --data-dir CLI flag
        config_value: Value from config file's data_dir key

    Returns:
        Description string like "CLI argument (--data-dir)" or "default"
    """
    if cli_arg is not None:
        return "CLI argument (--data-dir)"

    env_value = os.environ.get(ENV_VAR_NAME)
    if env_value:
        return f"environment variable ({ENV_VAR_NAME})"

    if config_value is not None:
        return "config file (data_dir)"

    return "default"
