import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from retrocast import __version__
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route
from retrocast.models.provenance import FileInfo, Manifest

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """
    Content types for manifest hashing.

    Usage guidelines:
    - BENCHMARK: Use when hashing a BenchmarkSet definition (benchmark.json.gz).
      The hash reflects all targets and their acceptable routes.

    - PREDICTIONS: Use ONLY after ingestion when you have a dict[str, list[Route]].
      This hashes Route objects (Pydantic models). Do NOT use during raw model
      execution (use "unknown" instead).

    - STOCK: Use when hashing a stock dictionary mapping InChIKey -> SMILES.

    - UNKNOWN: Use for raw model outputs (dict of raw predictions before ingestion).
      Content hashing is skipped; only file hash is computed. This prevents errors
      when trying to hash raw dict predictions as if they were Route objects.
    """

    BENCHMARK = "benchmark"
    PREDICTIONS = "predictions"
    STOCK = "stock"
    UNKNOWN = "unknown"


ContentTypeHint = Literal["benchmark", "predictions", "stock", "unknown"]


def calculate_file_hash(path: Path) -> str:
    """Computes SHA256 hash of a physical file in chunks."""
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return "error-hashing-file"


def generate_model_hash(model_name: str) -> str:
    """
    Generates a stable identifier for a model.
    Used for anonymized directory structures.
    """
    name_bytes = model_name.encode("utf-8")
    full_hash = hashlib.sha256(name_bytes).hexdigest()
    return f"retrocasted-model-{full_hash[:8]}"


def _calculate_benchmark_content_hash(benchmark: BenchmarkSet) -> str:
    """
    Internal: Hash a BenchmarkSet definition.

    The hash includes all acceptable routes for each target, ensuring that
    changes to the acceptable routes list will result in a different hash.
    """
    target_hashes = []
    for t in benchmark.targets.values():
        # We need to build a string that represents the target uniquely and deterministically.

        # 1. Basic fields
        # Use a separator that won't appear in identifiers
        parts = [t.id, t.smiles]

        # 2. Metadata (sort keys)
        if t.metadata:
            parts.append(json.dumps(t.metadata, sort_keys=True))
        else:
            parts.append("")

        # 3. Acceptable Routes
        # Hash all acceptable routes in order (order matters - first is primary)
        if t.acceptable_routes:
            route_hashes = [route.get_content_hash() for route in t.acceptable_routes]
            parts.append("|".join(route_hashes))
        else:
            parts.append("None")

        # Combine and hash this target
        target_str = "|".join(parts)
        target_hashes.append(hashlib.sha256(target_str.encode()).hexdigest())

    # Sort the list of target hashes (makes the set order irrelevant)
    target_hashes.sort()

    # Hash the combined list
    combined = "".join(target_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def _calculate_predictions_content_hash(routes: dict[str, list[Route]]) -> str:
    """
    Internal: Hash a dictionary of predicted routes.
    Ported from your old utils/hashing.py.
    """
    sorted_ids = sorted(routes.keys())
    route_hashes = []

    for target_id in sorted_ids:
        # Sort routes by rank for determinism within each target
        # We assume routes have a 'rank' attribute or we rely on list order if rank is missing
        # Using list order is safer if rank isn't guaranteed unique, but let's try rank first
        target_routes = routes[target_id]

        # Stable sort: try rank, fallback to signature
        try:
            sorted_routes = sorted(target_routes, key=lambda r: (r.rank, r.get_content_hash()))
        except AttributeError:
            # If rank is missing/None, sort purely by content signature
            sorted_routes = sorted(target_routes, key=lambda r: r.get_content_hash())

        for route in sorted_routes:
            r_hash = route.get_content_hash()  # This method exists on your Route model
            route_hashes.append(f"{target_id}:{r_hash}")

    combined = "".join(route_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def _calculate_stock_content_hash(stock: dict[str, str]) -> str:
    """
    Internal: Hash a stock dictionary (InChIKey -> SMILES).

    Only considers InChI keys (not SMILES) since multiple SMILES
    can represent the same molecule (tautomers, etc.).

    Args:
        stock: dict mapping InChIKey -> canonical SMILES

    Returns:
        SHA256 hash of sorted InChI keys
    """
    # Sort InChI keys for deterministic order
    sorted_inchi_keys = sorted(stock.keys())

    # Concatenate all InChI keys
    combined = "".join(sorted_inchi_keys)

    # Hash the result
    return hashlib.sha256(combined.encode()).hexdigest()


def create_manifest(
    action: str,
    sources: list[Path],
    outputs: list[tuple[Path, Any, ContentType | ContentTypeHint]],
    root_dir: Path,
    parameters: dict[str, Any] | None = None,
    statistics: dict[str, Any] | None = None,
) -> Manifest:
    """
    Generates a Manifest object with explicit content type specification.

    Args:
        action: Name of the action that produced these outputs
        sources: Input file paths
        outputs: List of (path, content_object, content_type) tuples
        root_dir: Root directory for relative path calculation
        parameters: Action parameters to record
        statistics: Action statistics to record

    Returns:
        Manifest object with file hashes and content hashes
    """
    logger.info("Generating manifest...")

    # Dispatch table for content hashing
    _HASH_DISPATCH = {
        ContentType.BENCHMARK: _calculate_benchmark_content_hash,
        ContentType.PREDICTIONS: _calculate_predictions_content_hash,
        ContentType.STOCK: _calculate_stock_content_hash,
        ContentType.UNKNOWN: lambda _: None,
    }

    def _get_relative_path(p: Path) -> str:
        try:
            return str(p.relative_to(root_dir))
        except ValueError:
            logger.warning(f"Path {p} is not inside root {root_dir}. Storing absolute path.")
            return str(p.resolve())

    source_infos = []
    for p in sources:
        if p.exists():
            source_infos.append(FileInfo(path=_get_relative_path(p), file_hash=calculate_file_hash(p)))
        else:
            logger.debug(f"Manifest source path not found on disk: {p}")

    output_infos = []
    for path, obj, content_type in outputs:
        f_hash = "file-not-written"
        if path.exists():
            f_hash = calculate_file_hash(path)

        # Explicit content hashing via dispatch table
        if isinstance(content_type, str):
            content_type = ContentType(content_type)

        hash_fn = _HASH_DISPATCH.get(content_type)
        if hash_fn is None:
            raise ValueError(f"Unknown content type: {content_type}")

        c_hash = hash_fn(obj) if content_type != ContentType.UNKNOWN else None

        output_infos.append(FileInfo(path=_get_relative_path(path), file_hash=f_hash, content_hash=c_hash))

    return Manifest(
        retrocast_version=__version__,
        created_at=datetime.now(UTC),
        action=action,
        parameters=parameters or {},
        source_files=source_infos,
        output_files=output_infos,
        statistics=statistics or {},
    )
