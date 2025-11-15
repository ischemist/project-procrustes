"""
Curation utilities for preparing and managing route datasets.

This module provides functions for creating manifests and managing
metadata for curated route collections.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retrocast.schemas import Route, _get_retrocast_version
from retrocast.utils.hashing import compute_routes_content_hash, generate_file_hash


def create_manifest(
    dataset_name: str,
    source_file: Path,
    output_file: Path,
    routes: dict[str, list[Route]],
    statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a manifest with file hashes, statistics, and metadata.

    This function generates a standardized manifest dictionary containing
    provenance information for a curated route dataset.

    Args:
        dataset_name: Name identifier for the dataset.
        source_file: Path to the source/input file.
        output_file: Path to the output file (must exist for hash computation).
        routes: Dictionary mapping target IDs to lists of Route objects.
        statistics: Optional custom statistics dictionary. If not provided,
                   basic statistics (n_routes_source, n_routes_saved) will be
                   computed from the routes.

    Returns:
        Dictionary containing the manifest with fields:
        - dataset: Dataset name
        - source_file: Name of source file
        - source_file_hash: SHA256 hash of source file
        - output_file: Name of output file
        - output_file_hash: SHA256 hash of output file
        - output_content_hash: Content hash of the routes
        - statistics: Statistics about the conversion/curation
        - timestamp: ISO format timestamp of creation
        - retrocast_version: Version of retrocast used
    """
    # Compute basic statistics if not provided
    if statistics is None:
        statistics = {
            "n_targets": len(routes),
            "n_routes_saved": sum(len(r) for r in routes.values()),
        }

    manifest = {
        "dataset": dataset_name,
        "source_file": source_file.name,
        "source_file_hash": generate_file_hash(source_file),
        "output_file": output_file.name,
        "output_file_hash": generate_file_hash(output_file),
        "output_content_hash": compute_routes_content_hash(routes),
        "statistics": statistics,
        "timestamp": datetime.now(UTC).isoformat(),
        "retrocast_version": _get_retrocast_version(),
    }

    return manifest
