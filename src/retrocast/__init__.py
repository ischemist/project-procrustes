"""
retrocast: A unified toolkit for retrosynthesis benchmark analysis.
"""

from importlib.metadata import PackageNotFoundError, version

from retrocast.adapters import ADAPTER_MAP, adapt_routes, adapt_single_route, get_adapter
from retrocast.curation.filtering import deduplicate_routes
from retrocast.curation.sampling import sample_k_by_length, sample_random_k, sample_top_k
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetInput

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
    "sample_k_by_length",
]
