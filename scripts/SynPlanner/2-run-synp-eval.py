"""
Run Synplanner MCTS retrosynthesis predictions on a batch of targets.

This script processes targets from a CSV file using Synplanner's MCTS algorithm
and saves results in a structured format similar to the DMS predictions script.

Example usage:
    uv run --extra synplanner --extra torch-cpu scripts/SynPlanner/2-run-synp-eval.py --target-name "ursa-expert-100"
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from synplan.chem.reaction_routes.io import make_json
from synplan.chem.reaction_routes.route_cgr import extract_reactions
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.evaluation import ValueNetworkFunction
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.mcts.tree import Tree, TreeConfig
from synplan.utils.config import PolicyNetworkConfig
from synplan.utils.loading import load_reaction_rules
from tqdm import tqdm

from ursa.io import load_targets_csv, save_json_gz
from ursa.utils.logging import logger

# establish a reliable base directory for pathing
base_dir = Path(__file__).resolve().parents[2]


def run_synplanner_predictions() -> None:
    """orchestrates the synplanner mcts predictions for a batch of targets."""
    # fmt:off
    parser = argparse.ArgumentParser(description="run synplanner mcts retrosynthesis predictions.")
    parser.add_argument("--target-name", type=str, required=True, help="name of the target set (e.g., 'rs-first-2').")
    parser.add_argument("--effort", type=str, choices=["standard", "high"], default="standard", help="effort level: 'standard' uses search-config.yaml, 'high' uses search-config-high.yaml (default: standard).")
    args = parser.parse_args()
    # fmt:on

    # --- configuration and setup ---
    model_data_dir = base_dir / "data" / "models" / "synplanner"
    config_filename = "search-eval-config-high.yaml" if args.effort == "high" else "search-eval-config.yaml"
    logger.info(f"using config: {config_filename}")
    config_path = model_data_dir / config_filename
    value_network_path = model_data_dir / "uspto" / "weights" / "value_network.ckpt"
    rank_weights = model_data_dir / "uspto" / "weights" / "ranking_policy_network.ckpt"
    reaction_rules_path = model_data_dir / "uspto" / "uspto_reaction_rules.pickle"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    search_config = {**config["tree"], **config["node_evaluation"]}
    policy_config = PolicyNetworkConfig.from_dict({**config["node_expansion"], **{"weights_path": rank_weights}})

    # --- load resources ---
    targets = load_targets_csv(base_dir / "data" / "targets" / f"{args.target_name}.csv")
    policy_function = PolicyNetworkFunction(policy_config=policy_config)
    reaction_rules = load_reaction_rules(reaction_rules_path)
    with open(base_dir / "data" / "models" / "assets" / "ursa-bb-stock-v3-canon.csv") as f:
        building_blocks = set(f.read().splitlines())

    if search_config["evaluation_type"] == "gcn" and value_network_path.exists():
        value_function = ValueNetworkFunction(weights_path=str(value_network_path))
    else:
        value_function = None

    tree_config = TreeConfig.from_dict(search_config)

    # --- output setup ---
    model_dir = "synplanner-eval-high" if args.effort == "high" else "synplanner-eval"
    save_dir = base_dir / "data" / "evaluations" / model_dir / args.target_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- processing loop ---
    results: dict[str, list[dict[Any, Any]]] = {}
    solved_count = 0
    start_time = time.time()

    pbar = tqdm(targets.items(), desc=f"processing {args.target_name}", unit="target")
    for target_key, target_smiles in pbar:
        try:
            target_mol = mol_from_smiles(target_smiles)
            if not target_mol:
                print(f"warning: could not create molecule for target {target_key}. skipping.")
                results[target_key] = []
                continue

            search_tree = Tree(
                target=target_mol,
                config=tree_config,
                reaction_rules=reaction_rules,
                building_blocks=building_blocks,
                expansion_function=policy_function,
                evaluation_function=value_function,
            )

            # run the search
            _ = list(search_tree)

            if bool(search_tree.winning_nodes):
                # the format synplanner returns is a bit weird. it's a dict where keys are internal ids.
                # these routes are already json-serializable dicts.
                raw_routes = make_json(extract_reactions(search_tree))
                # we wrap this in a list to match the format of other models.
                results[target_key] = list(raw_routes.values())
                solved_count += 1
            else:
                results[target_key] = []

        except Exception as e:
            print(f"error processing target {target_key}: {e}")
            results[target_key] = []

    end_time = time.time()

    # --- save results and summary ---
    summary = {
        "solved_count": solved_count,
        "total_targets": len(targets),
        "time_elapsed_seconds": round(end_time - start_time, 2),
    }

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_json_gz(results, save_dir / "results.json.gz")

    print(f"\n--- completed processing {len(targets)} targets ---")
    print(f"solved: {solved_count}/{len(targets)}")
    print(f"time elapsed: {summary['time_elapsed_seconds']:.2f} seconds")
    print(f"results saved to: {save_dir}")


if __name__ == "__main__":
    run_synplanner_predictions()
