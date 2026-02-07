"""
Run Retro* retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Retro* algorithm
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --directory scripts/planning/run-retrostar 2-run-og-retro-star.py --benchmark mkt-cnv-160 --max-routes 2
    uv run --directory scripts/planning/run-retrostar 2-run-og-retro-star.py --benchmark mkt-cnv-160 --effort high

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/retrocast/2-raw/retro-star[-{effort}][-max={N}]/{benchmark_name}/
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from retro_star.api import RSPlanner
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.paths import get_paths
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "retrocast"
paths = get_paths(DATA_DIR)

RETROSTAR_DIR = DATA_DIR / "0-assets" / "model-configs" / "retro-star"

# Retro* version - update when upgrading the library
PLANNER_VERSION = "1.1.0-dev"


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
    parser.add_argument(
        "--max-routes",
        type=int,
        default=1,
        help="Maximum number of routes to extract per target (default: 1). "
        "Values >1 will continue search after first route is found.",
    )
    args = parser.parse_args()

    # Set iterations based on effort level
    iterations = 500 if args.effort == "high" else 100

    # 1. Load Benchmark
    bench_path = paths["benchmarks"] / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {args.benchmark}"

    stock_path = paths["stocks"] / f"{benchmark.stock_name}.txt.gz"

    # 3. Setup Output
    folder_parts = ["retro-star"]
    if args.effort != "normal":
        folder_parts.append(args.effort)
    if args.max_routes > 1:
        folder_parts.append(f"max={args.max_routes}")
    folder_name = "-".join(folder_parts)
    save_dir = paths["raw"] / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")

    logger.info(f"effort: {args.effort} (iterations={iterations})")
    logger.info(f"max_routes: {args.max_routes}")

    # Initialize planner with the specified stock and iterations
    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=iterations,
        expansion_topk=50,
        starting_molecules=str(stock_path),
        mlp_templates=str(RETROSTAR_DIR / "one_step_model" / "template_rules_1.dat"),
        mlp_model_dump=str(RETROSTAR_DIR / "one_step_model" / "saved_rollout_state_1_2048.ckpt"),
        save_folder=str(RETROSTAR_DIR / "saved_models"),
        max_routes=args.max_routes,
    )

    logger.info("Retrosynthesis starting")

    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()
    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                result = planner.plan(target.smiles)

                if result and result["succ"]:
                    # Convert numpy types to native python types for JSON serialization
                    results[target.id] = convert_numpy(result)
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

    parameters = {
        "effort": args.effort,
        "max_routes": args.max_routes,
        "iterations": iterations,
        "expansion_topk": 50,
        "use_value_fn": True,
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action="scripts/planning/run-retrostar/2-run-og-retro-star.py",
        sources=[bench_path, stock_path],
        root_dir=DATA_DIR,
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        parameters=parameters,
        statistics=summary,
        directives={
            "adapter": "retrostar",
            "planner_version": PLANNER_VERSION,
            "raw_results_filename": "results.json.gz",
        },
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
