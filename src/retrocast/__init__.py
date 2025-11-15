"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from importlib.metadata import PackageNotFoundError, version

from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.domain.tree import deduplicate_routes, sample_k_by_depth, sample_random_k, sample_top_k
from retrocast.schemas import Molecule, ReactionStep, Route, TargetInput
from retrocast.utils.logging import setup_logging

setup_logging()

try:
    __version__ = version("retrocast")
except PackageNotFoundError:
    # Package not installed (running from source without editable install)
    __version__ = "0.0.0.dev0+unknown"
__all__ = [
    # Core schemas
    "Route",
    "Molecule",
    "ReactionStep",
    "TargetInput",
    # Adapter functions
    "adapt_single_route",
    "adapt_routes",
    "get_adapter",
    "ADAPTER_MAP",
    # Route processing utilities
    "deduplicate_routes",
    "sample_top_k",
    "sample_random_k",
    "sample_k_by_depth",
]
