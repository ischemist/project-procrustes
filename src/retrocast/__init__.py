"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from retrocast._version import __version__
from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.io import iter_route_corpus, load_route_corpus, save_route_corpus
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetInput
from retrocast.models.collections import CollectedBenchmarkRoutes
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.provenance import FileInfo, Manifest
from retrocast.models.stats import ModelStatistics
from retrocast.workflow.adapt import adapt_route_corpus
from retrocast.workflow.collect import collect_benchmark_predictions

__all__ = [
    "__version__",
    # Core schemas
    "Route",
    "Molecule",
    "ReactionStep",
    "TargetInput",
    "CollectedBenchmarkRoutes",
    # Workflow Schemas
    "BenchmarkSet",
    "EvaluationResults",
    "ModelStatistics",
    "FileInfo",
    "Manifest",
    # Explicit adaptation / collection workflow
    "adapt_route_corpus",
    "collect_benchmark_predictions",
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
    # Route-corpus io
    "save_route_corpus",
    "load_route_corpus",
    "iter_route_corpus",
]
