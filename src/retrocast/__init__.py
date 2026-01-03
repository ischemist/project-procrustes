"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetInput
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.models.stats import ModelStatistics


def _normalize_version_with_patch(version_str: str) -> str:
    """Ensure version always has explicit major.minor.micro format (e.g., 0.3.0.dev16 not 0.3.dev16)."""
    v = Version(version_str)
    # Reconstruct with explicit patch version
    base = f"{v.major}.{v.minor}.{v.micro}"

    # Add pre-release, post-release, dev, local parts if present
    parts = [base]
    if v.pre:
        parts.append(f"{v.pre[0]}{v.pre[1]}")
    if v.post is not None:
        parts.append(f".post{v.post}")
    if v.dev is not None:
        parts.append(f".dev{v.dev}")
    if v.local:
        parts.append(f"+{v.local}")

    return "".join(parts)


try:
    __version__ = _normalize_version_with_patch(version("retrocast"))
except PackageNotFoundError:
    # Package not installed (running from source without editable install)
    __version__ = "0.0.0.dev0+unknown"
__all__ = [
    # Core schemas
    "Route",
    "Molecule",
    "ReactionStep",
    "TargetInput",
    # Workflow Schemas
    "BenchmarkSet",
    "EvaluationResults",
    "ModelStatistics",
    "FileInfo",
    "Manifest",
    # Adapter functions
    "adapt_single_route",
    "adapt_routes",
    "get_adapter",
    "ADAPTER_MAP",
    # Route processing utilities
    "deduplicate_routes",
    "sample_top_k",
    "sample_random_k",
    "sample_k_by_length",
]
