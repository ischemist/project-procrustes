"""
Curation utilities for preparing and managing route datasets.

This module provides functions for creating manifests and managing
metadata for curated route collections.
"""

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retrocast.domain.tree import excise_reactions_from_route
from retrocast.models.chem import ReactionSignature, Route, _get_retrocast_version
from retrocast.utils.hashing import compute_routes_content_hash, generate_file_hash


def get_route_signatures(routes: dict[str, list[Route]]) -> set[str]:
    """Extract all route signatures from routes dict."""
    signatures = set()
    for route_list in routes.values():
        for route in route_list:
            signatures.add(route.get_signature())
    return signatures


def get_reaction_signatures(routes: dict[str, list[Route]]) -> set[ReactionSignature]:
    """Extract all reaction signatures from routes dict."""
    signatures: set[ReactionSignature] = set()
    for route_list in routes.values():
        for route in route_list:
            signatures.update(route.get_reaction_signatures())
    return signatures


def filter_routes_by_signature(routes: dict[str, list[Route]], exclude_signatures: set[str]) -> dict[str, list[Route]]:
    """Remove routes whose signatures are in exclude set."""
    filtered = {}
    for target_id, route_list in routes.items():
        kept_routes = [r for r in route_list if r.get_signature() not in exclude_signatures]
        if kept_routes:
            filtered[target_id] = kept_routes
    return filtered


def filter_routes_by_reaction_overlap(
    routes: dict[str, list[Route]], exclude_reactions: set[ReactionSignature]
) -> dict[str, list[Route]]:
    """Remove routes that have any overlapping reactions with exclude set."""
    filtered = {}
    for target_id, route_list in routes.items():
        kept_routes = []
        for route in route_list:
            route_reactions = route.get_reaction_signatures()
            if not route_reactions & exclude_reactions:  # No overlap
                kept_routes.append(route)
        if kept_routes:
            filtered[target_id] = kept_routes
    return filtered


def excise_reactions_from_routes(
    routes: dict[str, list[Route]], exclude_reactions: set[ReactionSignature]
) -> dict[str, list[Route]]:
    """
    Excise specific reactions from routes, keeping valid sub-routes.

    Unlike filter_routes_by_reaction_overlap which removes entire routes,
    this function removes only the overlapping reactions and retains any
    valid sub-routes that remain.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        exclude_reactions: Set of ReactionSignatures to excise from routes.

    Returns:
        Dictionary mapping target IDs to lists of Route objects with
        overlapping reactions excised. Routes with no remaining reactions
        are excluded. New sub-routes created from excision are included
        under their original target ID.
    """
    result: dict[str, list[Route]] = {}

    for target_id, route_list in routes.items():
        kept_routes: list[Route] = []
        for route in route_list:
            # Check if route has any overlapping reactions
            route_reactions = route.get_reaction_signatures()
            if not route_reactions & exclude_reactions:
                # No overlap, keep route as-is
                kept_routes.append(route)
            else:
                # Excise overlapping reactions and collect sub-routes
                sub_routes = excise_reactions_from_route(route, exclude_reactions)
                kept_routes.extend(sub_routes)

        if kept_routes:
            result[target_id] = kept_routes

    return result


def split_routes(
    routes: dict[str, list[Route]], train_ratio: float, seed: int
) -> tuple[dict[str, list[Route]], dict[str, list[Route]]]:
    """Split routes dict into train/val partitions."""
    target_ids = list(routes.keys())
    random.seed(seed)
    random.shuffle(target_ids)

    split_idx = int(len(target_ids) * train_ratio)
    train_ids = target_ids[:split_idx]
    val_ids = target_ids[split_idx:]

    train_routes = {tid: routes[tid] for tid in train_ids}
    val_routes = {tid: routes[tid] for tid in val_ids}

    return train_routes, val_routes


def _flatten_routes(routes: dict[str, list[Route]]) -> list[tuple[str, Route]]:
    """Flatten routes dict into list of (target_id, route) tuples."""
    flat = []
    for target_id, route_list in routes.items():
        for route in route_list:
            flat.append((target_id, route))
    return flat


def _unflatten_routes(flat_routes: list[tuple[str, Route]]) -> dict[str, list[Route]]:
    """Convert list of (target_id, route) tuples back to routes dict."""
    result: dict[str, list[Route]] = {}
    for target_id, route in flat_routes:
        if target_id not in result:
            result[target_id] = []
        result[target_id].append(route)
    return result


def sample_routes_random(routes: dict[str, list[Route]], n: int, seed: int) -> dict[str, list[Route]]:
    """
    Randomly sample n routes from the collection.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        n: Number of routes to sample.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with sampled routes, preserving target_id structure.

    Raises:
        ValueError: If n exceeds total number of available routes.
    """
    flat = _flatten_routes(routes)
    if n > len(flat):
        raise ValueError(f"Cannot sample {n} routes from {len(flat)} available")

    random.seed(seed)
    sampled = random.sample(flat, n)
    return _unflatten_routes(sampled)


def sample_routes_by_length(
    routes: dict[str, list[Route]], length_counts: dict[int, int], seed: int
) -> dict[str, list[Route]]:
    """
    Sample specific number of routes for each route length (depth).

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        length_counts: Dictionary mapping route length to number of routes to sample.
                      e.g., {2: 25, 4: 25, 6: 25, 8: 25}
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with sampled routes, preserving target_id structure.

    Raises:
        ValueError: If requested count exceeds available routes for any length.
    """
    # Group routes by depth
    by_depth: dict[int, list[tuple[str, Route]]] = {}
    flat = _flatten_routes(routes)
    for target_id, route in flat:
        depth = route.length
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append((target_id, route))

    # Sample from each depth
    sampled: list[tuple[str, Route]] = []
    random.seed(seed)

    for length, count in length_counts.items():
        available = by_depth.get(length, [])
        if count > len(available):
            raise ValueError(f"Cannot sample {count} routes of length {length}, only {len(available)} available")
        sampled.extend(random.sample(available, count))

    return _unflatten_routes(sampled)


def filter_routes_by_convergence(routes: dict[str, list[Route]], keep_convergent: bool) -> dict[str, list[Route]]:
    """
    Filter routes by whether they contain convergent reactions.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        keep_convergent: If True, keep only convergent routes. If False, keep only linear routes.

    Returns:
        Dictionary with filtered routes.
    """
    result: dict[str, list[Route]] = {}
    for target_id, route_list in routes.items():
        kept = [r for r in route_list if r.has_convergent_reaction == keep_convergent]
        if kept:
            result[target_id] = kept
    return result


def merge_routes(*route_dicts: dict[str, list[Route]]) -> dict[str, list[Route]]:
    """
    Merge multiple route dictionaries into one.

    Routes are combined by target_id. If the same target appears in multiple
    dictionaries, all routes are combined into a single list.

    Args:
        *route_dicts: Variable number of route dictionaries to merge.

    Returns:
        Merged dictionary of routes.
    """
    result: dict[str, list[Route]] = {}
    for routes in route_dicts:
        for target_id, route_list in routes.items():
            if target_id not in result:
                result[target_id] = []
            result[target_id].extend(route_list)
    return result


def create_manifest(
    dataset_name: str,
    source_file: Path,
    output_file: Path,
    routes: dict[str, list[Route]],
    statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a manifest with file hashes, statistics, and metadata.

    This function generates a standardized manifest dictionary containing
    provenance information for a curated route dataset.

    Args:
        dataset_name: Name identifier for the dataset.
        source_file: Path to the source/input file.
        output_file: Path to the output file (must exist for hash computation).
        routes: Dictionary mapping target IDs to lists of Route objects.
        statistics: Optional custom statistics dictionary. If not provided,
                   basic statistics (n_routes_source, n_routes_saved) will be
                   computed from the routes.

    Returns:
        Dictionary containing the manifest with fields:
        - dataset: Dataset name
        - source_file: Name of source file
        - source_file_hash: SHA256 hash of source file
        - output_file: Name of output file
        - output_file_hash: SHA256 hash of output file
        - output_content_hash: Content hash of the routes
        - statistics: Statistics about the conversion/curation
        - timestamp: ISO format timestamp of creation
        - retrocast_version: Version of retrocast used
    """
    # Compute basic statistics if not provided
    if statistics is None:
        statistics = {
            "n_targets": len(routes),
            "n_routes_saved": sum(len(r) for r in routes.values()),
        }

    manifest = {
        "dataset": dataset_name,
        "source_file": source_file.name,
        "source_file_hash": generate_file_hash(source_file),
        "output_file": output_file.name,
        "output_file_hash": generate_file_hash(output_file),
        "output_content_hash": compute_routes_content_hash(routes),
        "statistics": statistics,
        "timestamp": datetime.now(UTC).isoformat(),
        "retrocast_version": _get_retrocast_version(),
    }

    return manifest
