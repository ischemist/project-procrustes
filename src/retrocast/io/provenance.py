import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from retrocast import __version__
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import PredictedRoute, Route
from retrocast.models.provenance import FileInfo, Manifest

logger = logging.getLogger(__name__)


class ContentType(StrEnum):
    """
    Content types for manifest hashing.

    Usage guidelines:
    - BENCHMARK: Use when hashing a BenchmarkSet definition (benchmark.json.gz).
      The hash reflects all targets and their acceptable routes.

    - PREDICTIONS: Use ONLY after ingestion when you have a dict[str, list[Route]].
      This hashes Route objects (Pydantic models). Do NOT use during raw model
      execution (use "unknown" instead).

    - ROUTE_CORPUS: Use for route-corpus.jsonl.gz artifacts
      (ordered list[PredictedRoute] or legacy list[Route]). This preserves
      encounter order rather than target-keyed grouping.

    - STOCK: Use when hashing a stock dictionary mapping InChIKey -> SMILES.

    - UNKNOWN: Use for raw model outputs (dict of raw predictions before ingestion).
      Content hashing is skipped; only file hash is computed. This prevents errors
      when trying to hash raw dict predictions as if they were Route objects.
    """

    BENCHMARK = "benchmark"
    PREDICTIONS = "predictions"
    ROUTE_CORPUS = "route_corpus"
    STOCK = "stock"
    UNKNOWN = "unknown"


ContentTypeHint = Literal["benchmark", "predictions", "route_corpus", "stock", "unknown"]
UnlabeledManifestOutput: TypeAlias = tuple[Path, Any, ContentType | ContentTypeHint]
LabeledManifestOutput: TypeAlias = tuple[str, Path, Any, ContentType | ContentTypeHint]
ManifestOutput: TypeAlias = UnlabeledManifestOutput | LabeledManifestOutput


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
    Target buckets are order-invariant; route order within each bucket is
    preserved because list order is the canonical ranking signal.
    """
    sorted_ids = sorted(routes.keys())
    route_hashes = []

    for target_id in sorted_ids:
        for route in routes[target_id]:
            r_hash = route.get_content_hash()
            route_hashes.append(f"{target_id}:{r_hash}")

    combined = "".join(route_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def _calculate_predicted_route_content_hash(route: PredictedRoute) -> str:
    payload = route.model_dump(
        mode="json",
        exclude={"route": {"leaves", "length", "content_hash", "signature"}},
    )
    route_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(route_json.encode()).hexdigest()


def _calculate_route_corpus_content_hash(routes: list[Route | PredictedRoute]) -> str:
    """Internal: hash an ordered prediction route corpus."""
    route_hashes = [
        _calculate_predicted_route_content_hash(route)
        if isinstance(route, PredictedRoute)
        else route.get_content_hash()
        for route in routes
    ]
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


def _normalize_manifest_output(
    output: ManifestOutput,
) -> tuple[str | None, Path, Any, ContentType | ContentTypeHint]:
    items: tuple[Any, ...] = tuple(output)
    if len(items) == 3:
        path = items[0]
        obj = items[1]
        content_type = items[2]
        if not isinstance(path, Path):
            raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
        return None, path, obj, content_type

    if len(items) != 4:
        raise ValueError(f"manifest output must have 3 or 4 items, got {len(items)}")

    assert len(items) == 4
    label = items[0]
    path = items[1]
    obj = items[2]
    content_type = items[3]
    if not isinstance(label, str):
        raise TypeError(f"manifest output label must be str, got {type(label).__name__}")
    if not isinstance(path, Path):
        raise TypeError(f"manifest output path must be Path, got {type(path).__name__}")
    return label, path, obj, content_type


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
    """
    Generates a Manifest object with explicit content type specification.

    Args:
        action: Name of the action that produced these outputs
        sources: Input file paths
        outputs: List of (path, content_object, content_type) tuples
        root_dir: Root directory for relative path calculation
        parameters: Action parameters to record
        statistics: Action statistics to record
        directives: Directives for retrocast data consumption (e.g., adapter, raw_results_filename)
        summary: Optional richer action-specific summary payload
        release_name: Optional stable release identifier
        keyed_output_files: Store outputs as a dict keyed by label instead of a flat list

    Returns:
        Manifest object with file hashes and content hashes
    """
    logger.info("Generating manifest...")

    # Dispatch table for content hashing
    _HASH_DISPATCH = {
        ContentType.BENCHMARK: _calculate_benchmark_content_hash,
        ContentType.PREDICTIONS: _calculate_predictions_content_hash,
        ContentType.ROUTE_CORPUS: _calculate_route_corpus_content_hash,
        ContentType.STOCK: _calculate_stock_content_hash,
        ContentType.UNKNOWN: lambda _: None,
    }

    def _get_relative_path(p: Path) -> str:
        resolved_root = root_dir.resolve()
        resolved_path = p.resolve()
        try:
            return str(resolved_path.relative_to(resolved_root))
        except ValueError:
            logger.warning(f"Path {resolved_path} is not inside root {resolved_root}. Storing absolute path.")
            return str(resolved_path)

    source_infos = []
    for p in sources:
        if p.exists():
            source_infos.append(FileInfo(path=_get_relative_path(p), file_hash=calculate_file_hash(p)))
        else:
            logger.debug(f"Manifest source path not found on disk: {p}")

    output_infos: list[FileInfo] = []
    for output in outputs:
        label, path, obj, content_type = _normalize_manifest_output(output)
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

        output_infos.append(FileInfo(label=label, path=_get_relative_path(path), file_hash=f_hash, content_hash=c_hash))

    manifest_outputs: list[FileInfo] | dict[str, FileInfo]
    if keyed_output_files:
        manifest_outputs = {}
        for file_info in output_infos:
            if file_info.label is None:
                raise ValueError("keyed_output_files=True requires every output to include a label")
            manifest_outputs[file_info.label] = file_info
    else:
        manifest_outputs = output_infos

    return Manifest(
        retrocast_version=__version__,
        created_at=datetime.now(UTC),
        action=action,
        parameters=parameters or {},
        directives=directives or {},
        release_name=release_name,
        source_files=source_infos,
        output_files=manifest_outputs,
        statistics=statistics or {},
        summary=summary or {},
    )
