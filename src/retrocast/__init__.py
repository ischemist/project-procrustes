"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from retrocast._version import __version__
from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetInput
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.models.stats import ModelStatistics

__all__ = [
    "__version__",
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
