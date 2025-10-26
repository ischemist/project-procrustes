"""
Run AiZynthFinder Retro* retrosynthesis predictions on a batch of targets.

This script processes targets from a CSV file using AiZynthFinder's Retro* algorithm
and saves results in a structured format similar to the DMS predictions script.

Example usage:
    uv run --extra aizyn scripts/AiZynthFinder/4-run-aizyn-retro-star.py --target-name "ursa-bridge-100"

The target CSV file should be located at: data/{target_name}.csv
Results are saved to: data/evaluations/aizynthfinder-retro-star/{target_name}/
"""

import argparse
import json
import time
from pathlib import Path

from aizynthfinder.aizynthfinder import AiZynthFinder
from tqdm import tqdm

from ursa.io import load_targets_csv, save_json_gz

base_dir = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target")
    args = parser.parse_args()

    targets = load_targets_csv(base_dir / "data" / "targets" / f"{args.target_name}.csv")

    config_path = base_dir / "data" / "models" / "aizynthfinder" / "config_retrostar.yaml"

    save_dir = base_dir / "data" / "evaluations" / "aizynthfinder-retro-star" / args.target_name
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    solved_count = 0
    start = time.time()

    for target_key, target_smiles in tqdm(targets.items()):
        finder = AiZynthFinder(configfile=str(config_path))
        finder.stock.select("ursa-bb")
        finder.expansion_policy.select("uspto")
        finder.filter_policy.select("uspto")
        finder.target_smiles = target_smiles

        finder.tree_search()
        finder.build_routes()

        if finder.routes:
            routes_dict = finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
            results[target_key] = routes_dict
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
