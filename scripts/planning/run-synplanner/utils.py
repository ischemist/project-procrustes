"""Shared utilities for Synplanner scripts."""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

from synplan.utils.config import CombinedPolicyConfig, PolicyNetworkConfig
from synplan.utils.loading import load_building_blocks, load_combined_policy_function, load_policy_function


def load_policy_from_config(
    policy_params: dict,
    filtering_weights_path: str,
    ranking_weights_path: str,
) -> Callable:
    """Loads the appropriate policy function based on configuration.

    Args:
        policy_params: Dictionary containing policy configuration, including 'mode'
            ('ranking' or 'combined'), 'top_rules', and 'rule_prob_threshold'.
        filtering_weights_path: Path to the filtering policy network weights.
        ranking_weights_path: Path to the ranking policy network weights.

    Returns:
        The loaded policy function callable.
    """
    mode = policy_params.get("mode", "ranking")
    if mode == "combined":
        combined_policy_config = CombinedPolicyConfig(
            filtering_weights_path=filtering_weights_path,
            ranking_weights_path=ranking_weights_path,
            top_rules=policy_params.get("top_rules", 50),
            rule_prob_threshold=policy_params.get("rule_prob_threshold", 0.0),
        )
        return load_combined_policy_function(combined_config=combined_policy_config)
    # 'ranking' or other modes
    return load_policy_function(policy_config=PolicyNetworkConfig(weights_path=ranking_weights_path))


def load_building_blocks_cached(
    stock_path: Path,
    *,
    silent: bool = True,
) -> set[str]:
    """Load building blocks with caching for SynPlanner's standardization.

    SynPlanner uses special canonicalization that takes ~5 minutes for large stocks.
    This function checks for a pre-standardized cache file and uses it if available,
    otherwise standardizes and saves the result for future runs.

    Args:
        stock_path: Path to the original stock CSV file (e.g., buyables-stock.csv.gz).
        silent: Suppress progress output from load_building_blocks.

    Returns:
        Set of SMILES strings representing building blocks.
    """
    # Check for cached standardized version (e.g., buyables-stock-synplanner.csv.gz)
    cached_path = stock_path.with_name(stock_path.name.replace(".csv.gz", "-synplanner.csv.gz"))

    if cached_path.exists():
        return load_building_blocks(cached_path, standardize=False, silent=silent)

    # Load with standardization (slow ~5 min)
    building_blocks = load_building_blocks(stock_path, standardize=True, silent=silent)

    # Save cached version for next time
    with gzip.open(cached_path, "wt", encoding="utf-8") as f:
        f.write("SMILES\n")  # header
        for smiles in building_blocks:
            f.write(f"{smiles}\n")

    return building_blocks
