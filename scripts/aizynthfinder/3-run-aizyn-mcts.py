"""
Run AiZynthFinder MCTS retrosynthesis predictions on a batch of targets.

This script processes targets from a CSV file using AiZynthFinder's MCTS algorithm
and saves results in a structured format similar to the DMS predictions script.

Example usage:
    uv run --extra aizyn scripts/aizynthfinder/3-run-aizyn-mcts.py --benchmark uspto-190

The target CSV file should be located at: data/{target_name}.csv
Results are saved to: data/evaluations/aizynthfinder-mcts/{target_name}/

You might need to install some build tools to install aizynthfinder deps on a clean EC2 instance.
```bash
sudo apt-get update && sudo apt-get install build-essential python3.11-dev libxrender1
```

Also, cgrtools doesn't have any wheels on linux, so you have to do some yak shaving, thankfully uv is a godsend
```bash
# create a temp env to build the artifact
uv venv .build_tmp
source .build_tmp/bin/activate

# install the legacy build stack
uv pip install "cython<3" setuptools wheel

# build cgrtools without isolation (uses the old cython we just installed)
uv pip install cgrtools==4.1.35 --no-build-isolation

# now nuke the temp env and run sync
deactivate
rm -rf .build_tmp
uv sync --extra aizyn
```
"""

import argparse
import time
from pathlib import Path
from typing import Any

from aizynthfinder.aizynthfinder import AiZynthFinder
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import ExecutionStats
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Name of the benchmark set (e.g. stratified-linear-600)"
    )
    args = parser.parse_args()

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {args.benchmark}"

    # 2. Setup Output
    save_dir = BASE_DIR / "data" / "2-raw" / "aizynthfinder-mcts" / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = BASE_DIR / "data" / "0-assets" / "model-configs" / "aizynthfinder" / "config_mcts.yaml"

    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    runtime = ExecutionStats()

    for target in tqdm(benchmark.targets.values()):
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()

        finder = AiZynthFinder(configfile=str(config_path))
        finder.stock.select(benchmark.stock_name)
        finder.expansion_policy.select("uspto")
        finder.filter_policy.select("uspto")

        try:
            finder.target_smiles = target.smiles
            finder.tree_search()
            finder.build_routes()

            if finder.routes:
                routes_dict = finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
                results[target.id] = routes_dict
                solved_count += 1
            else:
                results[target.id] = {}
        except Exception as e:
            logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
            results[target.id] = {}
        finally:
            t_end_wall = time.perf_counter()
            t_end_cpu = time.process_time()
            runtime.wall_time[target.id] = t_end_wall - t_start_wall
            runtime.cpu_time[target.id] = t_end_cpu - t_start_cpu

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action="scripts/aizynthfinder/3-run-aizyn-mcts.py",
        sources=[bench_path, config_path],
        outputs=[(save_dir / "results.json.gz", results)],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
