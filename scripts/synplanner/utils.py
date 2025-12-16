"""Shared utilities for Synplanner scripts."""

from collections.abc import Callable

from synplan.utils.config import CombinedPolicyConfig, PolicyNetworkConfig
from synplan.utils.loading import load_combined_policy_function, load_policy_function


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
