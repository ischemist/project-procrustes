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


def clean_and_prioritize_pools(
    primary: list[BenchmarkTarget], secondary: list[BenchmarkTarget]
) -> tuple[list[BenchmarkTarget], list[BenchmarkTarget]]:
    """
    Cleans two pools of targets based on conflicts, but keeps them separate.

    Returns:
        (cleaned_primary, cleaned_secondary)
    """
    logger.info(f"Cleaning pools: Primary ({len(primary)}) + Secondary ({len(secondary)})")

    # 1. Filter Secondary by Route Signature (remove duplicates)
    primary_sigs = {t.ground_truth.get_signature() for t in primary if t.ground_truth}

    secondary_unique_routes = []
    for t in secondary:
        if t.ground_truth and t.ground_truth.get_signature() in primary_sigs:
            continue
        secondary_unique_routes.append(t)

    # 2. Identify Ambiguous SMILES
    primary_smiles = {t.smiles for t in primary}
    secondary_smiles = {t.smiles for t in secondary_unique_routes}
    ambiguous_smiles = primary_smiles.intersection(secondary_smiles)

    if ambiguous_smiles:
        logger.warning(f"  Removing {len(ambiguous_smiles)} ambiguous targets from BOTH pools.")

    # 3. Construct Final Lists
    clean_primary = [t for t in primary if t.smiles not in ambiguous_smiles]
    clean_secondary = [t for t in secondary_unique_routes if t.smiles not in ambiguous_smiles]

    logger.info(f"Cleaned sizes: Primary {len(clean_primary)}, Secondary {len(clean_secondary)}")

    return clean_primary, clean_secondary
