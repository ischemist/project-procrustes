"""Python-facing route generation backed by the Rust engine."""

from __future__ import annotations

from retrocast.models.route import Route
from retrocast.typing import InChIKeyStr


def generate_pruned_routes(route: Route, stock: set[InChIKeyStr]) -> list[Route]:
    """Generate stock-solvable route variants by pruning stock intermediates."""
    from retrocast import native

    return native.generate_pruned_routes(route, {str(key) for key in stock})
