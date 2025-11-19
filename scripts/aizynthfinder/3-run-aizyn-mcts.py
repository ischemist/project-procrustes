"""
Run AiZynthFinder MCTS retrosynthesis predictions on a batch of targets.

This script processes targets from a CSV file using AiZynthFinder's MCTS algorithm
and saves results in a structured format similar to the DMS predictions script.

Example usage:
    uv run --extra aizyn scripts/aizynthfinder/3-run-aizyn-mcts.py --benchmark random-n5-2-seed=20251030

The target CSV file should be located at: data/{target_name}.csv
Results are saved to: data/evaluations/aizynthfinder-mcts/{target_name}/

You might need to install some build tools to install aizynthfinder deps on a clean EC2 instance.
```bash
sudo apt-get update && sudo apt-get install build-essential python3.11-dev libxrender1
```
"""

import argparse
import time
from pathlib import Path
from typing import Any

from aizynthfinder.aizynthfinder import AiZynthFinder
from tqdm import tqdm

from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
base_dir = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Name of the benchmark set (e.g. stratified-linear-600)"
    )
    args = parser.parse_args()

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)

    # 2. Setup Output
    save_dir = BASE_DIR / "data" / "2-raw" / "aizynthfinder-mcts" / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "data" / "models" / "aizynthfinder" / "config_mcts.yaml"

    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    finder = AiZynthFinder(configfile=str(config_path))
    finder.stock.select("retrocast-bb")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    for target in tqdm(benchmark.targets.values()):
        finder.target_smiles = target.smiles

        finder.tree_search()
        finder.build_routes()

        if finder.routes:
            routes_dict = finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
            results[target.id] = routes_dict
            solved_count += 1
        else:
            results[target.id] = {}

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
        "wall_time_ms": end_wall - start_wall,
        "cpu_time_ms": end_cpu - start_cpu,
    }

    save_json_gz(results, save_dir / "results.json.gz")
    manifest = create_manifest(
        action="scripts/aizynthfinder/3-run-aizyn-mcts.py",
        sources=[bench_path, config_path],
        outputs=[(save_dir / "results.json.gz", results)],
        statistics=summary,
    )

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
