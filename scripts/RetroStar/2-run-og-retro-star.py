"""
Run Retro* retrosynthesis predictions on a batch of targets.

Example usage:
    uv run --extra retro-star --extra torch-cpu scripts/RetroStar/2-run-og-retro-star.py --target-name "ursa-expert-100"

The target CSV file should be located at: data/targets/{target_name}.csv
Results are saved to: data/evaluations/retro-star/{target_name}/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from retro_star.api import RSPlanner
from tqdm import tqdm

from ursa.io import load_targets_csv, save_json_gz

base_dir = Path(__file__).resolve().parents[2]


def convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target set")
    args = parser.parse_args()

    # Load targets
    target_file = base_dir / "data" / "targets" / f"{args.target_name}.csv"
    targets = load_targets_csv(target_file)

    # Setup save directory
    save_dir = base_dir / "data" / "evaluations" / "retro-star" / args.target_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize planner
    starting_molecules = base_dir / "data" / "models" / "assets" / "ursa-bb-stock-v3-canon.csv"
    retro_star_dir = base_dir / "data" / "models" / "retro-star"

    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=100,
        expansion_topk=50,
        starting_molecules=str(starting_molecules),
        mlp_templates=str(retro_star_dir / "one_step_model" / "template_rules_1.dat"),
        mlp_model_dump=str(retro_star_dir / "one_step_model" / "saved_rollout_state_1_2048.ckpt"),
        save_folder=str(retro_star_dir / "saved_models"),
    )

    results = {}
    solved_count = 0
    start = time.time()

    for target_key, target_smiles in tqdm(targets.items()):
        result = planner.plan(target_smiles)

        if result and result["succ"]:
            # Convert numpy types to native python types for JSON serialization
            results[target_key] = convert_numpy(result)
            solved_count += 1
        else:
            results[target_key] = {}

    end = time.time()

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
