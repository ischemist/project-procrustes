"""
Run AiZynthFinder Retro* retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using AiZynthFinder's Retro* algorithm
and saves results in a structured format similar to the MCTS predictions script.

Example usage:
    uv run --extra aizyn scripts/aizynthfinder/4-run-aizyn-retro-star.py --benchmark ref-cnv-400 --effort high
    uv run --extra aizyn scripts/aizynthfinder/4-run-aizyn-retro-star.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/aizynthfinder-retro-star[-{effort}]/{benchmark_name}/
"""

import argparse
from pathlib import Path
from typing import Any

from aizynthfinder.aizynthfinder import AiZynthFinder
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Name of the benchmark set (e.g. stratified-linear-600)"
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="normal",
        choices=["normal", "high"],
        help="Search effort level: normal (100 iterations) or high (500 iterations)",
    )
    args = parser.parse_args()

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {args.benchmark}"

    # 2. Setup Output
    folder_name = "aizynthfinder-retro-star" if args.effort == "normal" else f"aizynthfinder-retro-star-{args.effort}"
    save_dir = BASE_DIR / "data" / "2-raw" / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_suffix = "" if args.effort == "normal" else f"-{args.effort}"
    config_path = (
        BASE_DIR / "data" / "0-assets" / "model-configs" / "aizynthfinder" / f"config-retrostar{config_suffix}.yaml"
    )

    logger.info(f"effort: {args.effort} (config: {config_path.name})")

    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    for target in tqdm(benchmark.targets.values()):
        with timer.measure(target.id):
            try:
                finder = AiZynthFinder(configfile=str(config_path))
                finder.stock.select(benchmark.stock_name)
                finder.expansion_policy.select("uspto")
                finder.filter_policy.select("uspto")

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

    runtime = timer.to_model()

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action="scripts/aizynthfinder/4-run-aizyn-retro-star.py",
        sources=[bench_path, config_path],
        root_dir=BASE_DIR / "data",
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
