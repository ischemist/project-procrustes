import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retrocast import __version__
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Route
from retrocast.models.provenance import FileInfo, Manifest

logger = logging.getLogger(__name__)


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
    """
    target_hashes = []
    for t in benchmark.targets.values():
        # We need to build a string that represents the target uniquely and deterministically.

        # 1. Basic fields
        # Use a separator that won't appear in identifiers
        parts = [t.id, t.smiles, str(t.route_length), str(t.is_convergent)]

        # 2. Metadata (sort keys)
        if t.metadata:
            parts.append(json.dumps(t.metadata, sort_keys=True))
        else:
            parts.append("")

        # 3. Ground Truth Route
        # We use the route's OWN content hash method, which is designed to be safe
        # (it excludes 'leaves' and handles sorting).
        if t.ground_truth:
            parts.append(t.ground_truth.get_content_hash())
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


def create_manifest(
    action: str,
    sources: list[Path],
    outputs: list[tuple[Path, Any]],  # Tuple of (path, python_object)
    parameters: dict[str, Any] | None = None,
    statistics: dict[str, Any] | None = None,
) -> Manifest:
    """
    Generates a Manifest object. Automatically detects object type to calculate content hash.
    """
    logger.info("Generating manifest...")

    source_infos = []
    for p in sources:
        if p.exists():
            source_infos.append(FileInfo(path=p.name, file_hash=calculate_file_hash(p)))
        else:
            # Sometimes we pass a logical source that isn't a file (like a pickle stream)
            # In that case, just log it.
            logger.debug(f"Manifest source path not found on disk: {p}")

    output_infos = []
    for path, obj in outputs:
        f_hash = "file-not-written"
        if path.exists():
            f_hash = calculate_file_hash(path)

        # Polymorphic content hashing
        c_hash = None
        if isinstance(obj, BenchmarkSet):
            c_hash = _calculate_benchmark_content_hash(obj)
        elif isinstance(obj, dict) and len(obj) > 0:
            # Check if this is truly a dict[str, list[Route]]
            first_value = next(iter(obj.values()))
            if (
                isinstance(first_value, list)
                and len(first_value) > 0
                and all(isinstance(r, Route) for r in first_value)
            ):
                c_hash = _calculate_predictions_content_hash(obj)
            # Otherwise, fall through and only use file hash

        output_infos.append(FileInfo(path=path.name, file_hash=f_hash, content_hash=c_hash))

    return Manifest(
        retrocast_version=__version__,
        created_at=datetime.now(UTC),
        action=action,
        parameters=parameters or {},
        source_files=source_infos,
        output_files=output_infos,
        statistics=statistics or {},
    )
