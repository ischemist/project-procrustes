from typing import Literal

from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.utils.logging import logger

RouteType = Literal["linear", "convergent"]


def filter_by_route_type(benchmark: BenchmarkSet, route_type: RouteType) -> list[BenchmarkTarget]:
    """
    Extracts targets based on their synthesis topology.
    """
    if route_type == "convergent":
        return [t for t in benchmark.targets.values() if t.is_convergent]
    elif route_type == "linear":
        return [t for t in benchmark.targets.values() if not t.is_convergent]
    else:
        raise ValueError(f"Unknown route type: {route_type}")


def merge_target_pools(primary: list[BenchmarkTarget], secondary: list[BenchmarkTarget]) -> list[BenchmarkTarget]:
    """
    Merges two pools of targets with strict deduplication.

    Logic:
    1. Remove targets from Secondary if their Route Signature exists in Primary.
       (Prefer the Primary's version of the route).
    2. Identify targets (by SMILES) that exist in both the Primary and the
       (filtered) Secondary.
    3. Remove THESE ambiguous targets from BOTH lists.

    Result: A list where every SMILES is unique and has exactly one GT route.
    """
    logger.info(f"Merging pools: Primary ({len(primary)}) + Secondary ({len(secondary)})")

    # 1. Filter Secondary by Route Signature (remove duplicates)
    # We assume all targets in curation phase have ground_truth.
    primary_sigs = {t.ground_truth.get_signature() for t in primary if t.ground_truth}

    secondary_unique_routes = []
    routes_dropped = 0

    for t in secondary:
        if t.ground_truth and t.ground_truth.get_signature() in primary_sigs:
            routes_dropped += 1
        else:
            secondary_unique_routes.append(t)

    if routes_dropped:
        logger.info(f"  Dropped {routes_dropped} routes from Secondary (duplicate signature in Primary).")

    # 2. Identify Ambiguous SMILES (targets present in both)
    primary_smiles = {t.smiles for t in primary}
    secondary_smiles = {t.smiles for t in secondary_unique_routes}

    ambiguous_smiles = primary_smiles.intersection(secondary_smiles)

    if ambiguous_smiles:
        logger.warning(f"  Found {len(ambiguous_smiles)} targets present in both pools with DIFFERENT routes.")
        logger.warning("  Removing these targets from BOTH pools to ensure unique ground truth.")

    # 3. Construct Final List
    final_pool = []

    # Add non-ambiguous Primary
    p_added = 0
    for t in primary:
        if t.smiles not in ambiguous_smiles:
            final_pool.append(t)
            p_added += 1

    # Add non-ambiguous Secondary
    s_added = 0
    for t in secondary_unique_routes:
        if t.smiles not in ambiguous_smiles:
            final_pool.append(t)
            s_added += 1

    logger.info(f"Merge complete. Final pool size: {len(final_pool)} (Primary: {p_added}, Secondary: {s_added})")

    return final_pool
