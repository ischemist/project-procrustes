"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.domain.tree import deduplicate_routes, sample_k_by_depth, sample_random_k, sample_top_k
from retrocast.schemas import Molecule, ReactionStep, Route, TargetInput
from retrocast.utils.logging import setup_logging

setup_logging()

__version__ = "0.1.0"
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
