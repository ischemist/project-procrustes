from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from retrocast.exceptions import RetroCastException
from retrocast.utils.logging import logger

if TYPE_CHECKING:
    from retrocast.models.chem import Route


def generate_file_hash(path: Path) -> str:
    """Computes the sha256 hash of a file's content."""
    try:
        with path.open("rb") as f:
            file_bytes = f.read()
            return hashlib.sha256(file_bytes).hexdigest()
    except OSError as e:
        logger.error(f"Could not read file for hashing: {path}")
        raise RetroCastException(f"File I/O error on {path}: {e}") from e


def generate_model_hash(model_name: str) -> str:
    """
    Generates a short, deterministic, and stable hash for a model's name.

    This provides a consistent identifier for organizing outputs. We truncate
    a sha256 hash to 12 characters, which is sufficient to avoid collisions
    for any practical number of models.

    Returns:
        A 'retrocast-model-' prefixed, 12-character hex digest.
    """
    name_bytes = model_name.encode("utf-8")
    full_hash = hashlib.sha256(name_bytes).hexdigest()
    return f"retrocasted-model-{full_hash[:8]}"


def generate_source_hash(model_name: str, file_hashes: list[str]) -> str:
    """
    Generates a full, deterministic hash for a specific run based on the
    model's name and the exact content of all its output files.

    This is used for cryptographic proof of what data was processed, and is
    stored in the manifest, NOT in the filename.

    Args:
        model_name: The name of the model being processed.
        file_hashes: A sorted list of the sha256 hashes of all input files.

    Returns:
        A 'retrocast-source-' prefixed full sha256 hex digest.
    """
    sorted_hashes = sorted(file_hashes)
    run_signature = model_name + "".join(sorted_hashes)
    run_bytes = run_signature.encode("utf-8")
    hasher = hashlib.sha256(run_bytes)
    return f"retrocasted-source-{hasher.hexdigest()}"


def compute_routes_content_hash(routes: dict[str, list[Route]]) -> str:
    """
    Computes an order-agnostic hash of routes content.

    This hash is deterministic and will be identical for routes that contain
    the same data, regardless of insertion order. It uses Route.get_content_hash()
    which includes all route data (rank, metadata, tree structure, etc.).

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.

    Returns:
        A SHA256 hex digest representing the content hash.
    """
    # Sort by target_id for determinism
    sorted_target_ids = sorted(routes.keys())
    route_hashes = []

    for target_id in sorted_target_ids:
        # Sort routes by rank for determinism within each target
        for route in sorted(routes[target_id], key=lambda r: r.rank):
            route_hash = route.get_content_hash()
            route_hashes.append(f"{target_id}:{route_hash}")

    combined = "".join(route_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()
