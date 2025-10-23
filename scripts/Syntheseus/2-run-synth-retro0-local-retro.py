"""
Run Syntheseus LocalRetroModel retrosynthesis predictions on a batch of targets using RetroStar search.

This script processes targets from a CSV file using Syntheseus's LocalRetroModel with RetroStar search
and saves results in a structured format similar to the DMS and AiZynthFinder scripts.

Example usage:
    uv run --extra synth-knn scripts/Syntheseus/2-run-synth-retro0-local-retro.py --target-name "test-targets"

The target CSV file should be located at: data/targets/{target_name}.csv
Results are saved to: data/evaluations/syntheseus-retro0-local-retro/{target_name}/
"""

import argparse
import json
import time
from pathlib import Path

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator, ReactionModelLogProbCost
from tqdm import tqdm

from ursa.io import load_targets_csv, save_json_gz
from ursa.utils.serializers import serialize_route

base_dir = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target")
    args = parser.parse_args()

    # Load targets
    targets = load_targets_csv(base_dir / "data" / "targets" / f"{args.target_name}.csv")

    # Load building blocks
    with open(base_dir / "data" / "models" / "assets" / "ursa-bb-stock-v3-canon.csv") as f:
        building_blocks = [line.strip() for line in f if line.strip()]

    # Set up inventory with the building blocks
    inventory = SmilesListInventory(smiles_list=building_blocks)

    # Set up the reaction model
    model = LocalRetroModel(use_cache=True, default_num_results=10)

    # Create save directory
    save_dir = base_dir / "data" / "evaluations" / "syntheseus-retro0-local-retro" / args.target_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set up RetroStar cost functions and value function
    or_node_cost_fn = retro_star.MolIsPurchasableCost()  # type:ignore
    and_node_cost_fn = ReactionModelLogProbCost(normalize=False)
    retro_star_value_function = ConstantNodeEvaluator(0.0)

    results = {}
    solved_count = 0
    start = time.time()

    for target_key, target_smiles in tqdm(targets.items()):
        # Set up RetroStar search algorithm for each target
        search_algorithm = retro_star.RetroStarSearch(
            reaction_model=model,
            mol_inventory=inventory,
            or_node_cost_fn=or_node_cost_fn,
            and_node_cost_fn=and_node_cost_fn,
            value_function=retro_star_value_function,
            limit_reaction_model_calls=100,  # max number of model calls
            time_limit_s=300.0,  # max runtime in seconds (increased for RetroStar)
        )

        try:
            # Run search
            test_mol = Molecule(target_smiles)
            search_algorithm.reset()
            output_graph, _ = search_algorithm.run_from_mol(test_mol)

            # Extract routes using cost order (better for RetroStar)
            routes = list(iter_routes_cost_order(output_graph, max_routes=10))

            if routes:
                # Serialize all routes for this target
                serialized_routes = []
                for route in routes:
                    try:
                        serialized_route = serialize_route(route, target_smiles)
                        serialized_routes.append(serialized_route)
                    except Exception as e:
                        print(f"Warning: Could not serialize route for target {target_key}: {e}")

                if serialized_routes:
                    results[target_key] = serialized_routes
                    solved_count += 1
                else:
                    results[target_key] = []
            else:
                results[target_key] = []

        except Exception as e:
            print(f"Error processing target {target_key}: {e}")
            results[target_key] = []

    end = time.time()

    # Save summary
    summary = {
        "solved_count": solved_count,
        "total_targets": len(targets),
        "time_elapsed": end - start,
    }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f)
    save_json_gz(results, save_dir / "results.json.gz")

    print(f"Completed processing {len(targets)} targets")
    print(f"Solved: {solved_count}")
    print(f"Time elapsed: {end - start:.2f} seconds")
