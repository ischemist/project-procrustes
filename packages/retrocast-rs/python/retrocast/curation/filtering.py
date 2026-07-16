"""Python-facing curation helpers backed by the Rust engine."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Literal

from retrocast.models.route import Route
from retrocast.models.task import Benchmark, Target

RouteType = Literal["linear", "convergent"]


def excise_reactions_from_route(route: Route, exclude: set[str]) -> list[Route]:
    """Cut excluded reactions and return the non-empty Rust-built fragments."""
    from retrocast import native

    return native.excise_reactions(route, exclude)


def deduplicate_routes(
    routes: list[Route],
    *,
    key: Callable[[Route], Hashable] | None = None,
) -> list[Route]:
    """Remove duplicate route identities while preserving first occurrence.

    Default route identity is computed in Rust. A caller-supplied Python key is
    intentionally evaluated in Python, after which only index selection remains.
    """
    if key is None:
        from retrocast import native

        return native.deduplicate_routes(routes)

    seen: set[Hashable] = set()
    unique = []
    for route in routes:
        identity = key(route)
        if identity not in seen:
            seen.add(identity)
            unique.append(route)
    return unique


def filter_by_route_type(benchmark: Benchmark, route_type: RouteType) -> list[Target]:
    """Return targets whose primary acceptable route has the requested topology."""
    from retrocast import native

    return native.filter_by_route_type(benchmark, route_type)


def clean_and_prioritize_pools(primary: list[Target], secondary: list[Target]) -> tuple[list[Target], list[Target]]:
    """Remove route duplicates and ambiguous targets across prioritized pools."""
    from retrocast import native

    return native.clean_and_prioritize_pools(primary, secondary)


def route_is_convergent(route: Route) -> bool:
    """Return whether a reaction combines multiple synthesized branches."""
    from retrocast import native

    return native.route_is_convergent(route)
