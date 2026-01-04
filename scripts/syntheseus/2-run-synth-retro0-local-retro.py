"""
Run Syntheseus LocalRetroModel retrosynthesis predictions on a batch of targets using RetroStar search.

This script processes targets from a benchmark using Syntheseus's LocalRetroModel with RetroStar search
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --extra syntheseus scripts/syntheseus/2-run-synth-retro0-local-retro.py --benchmark mkt-lin-500
    uv run --extra syntheseus scripts/syntheseus/2-run-synth-retro0-local-retro.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/syntheseus-retro0-local-retro[-{effort}]/{benchmark_name}/
"""

import argparse
import gzip
from pathlib import Path
from typing import Any

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator, ReactionModelLogProbCost
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger
from retrocast.utils.serializers import serialize_route

BASE_DIR = Path(__file__).resolve().parents[2]

STOCKS_DIR = BASE_DIR / "data" / "1-benchmarks" / "stocks"

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
        help="Search effort level: normal or high",
    )
    args = parser.parse_args()

    iterations = 500 if args.effort == "high" else 100

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {args.benchmark}"

    # 2. Load Stock
    stock_path = STOCKS_DIR / f"{benchmark.stock_name}.txt.gz"
    with gzip.open(stock_path, "rt") as f:
        building_blocks = list(f.read().splitlines())

    # 3. Setup Output
    folder_name = (
        "syntheseus-retro0-local-retro" if args.effort == "normal" else f"syntheseus-retro0-local-retro-{args.effort}"
    )
    save_dir = BASE_DIR / "data" / "2-raw" / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort}")

    # 4. Set up inventory with the building blocks
    inventory = SmilesListInventory(smiles_list=building_blocks)

    # 5. Set up the reaction model
    model = LocalRetroModel(use_cache=True, default_num_results=10)

    # 6. Set up RetroStar cost functions and value function
    or_node_cost_fn = retro_star.MolIsPurchasableCost()  # type:ignore
    and_node_cost_fn = ReactionModelLogProbCost(normalize=False)
    retro_star_value_function = ConstantNodeEvaluator(0.0)

    # 7. Run Predictions
    logger.info("Retrosynthesis starting")

    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                # Set up RetroStar search algorithm for each target
                search_algorithm = retro_star.RetroStarSearch(
                    reaction_model=model,
                    mol_inventory=inventory,
                    or_node_cost_fn=or_node_cost_fn,
                    and_node_cost_fn=and_node_cost_fn,
                    value_function=retro_star_value_function,
                    limit_reaction_model_calls=iterations,  # max number of model calls
                    time_limit_s=300.0,  # max runtime in seconds (increased for RetroStar)
                )

                # Run search
                test_mol = Molecule(target.smiles)
                search_algorithm.reset()
                output_graph, _ = search_algorithm.run_from_mol(test_mol)

                # Extract routes using cost order (better for RetroStar)
                routes = list(iter_routes_cost_order(output_graph, max_routes=10))

                if routes:
                    # Serialize all routes for this target
                    serialized_routes = []
                    for route in routes:
                        try:
                            serialized_route = serialize_route(route, target.smiles)
                            serialized_routes.append(serialized_route)
                        except Exception as e:
                            logger.warning(f"Could not serialize route for target {target.id}: {e}")

                    if serialized_routes:
                        results[target.id] = serialized_routes
                        solved_count += 1
                    else:
                        results[target.id] = []
                else:
                    results[target.id] = []

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                results[target.id] = []

    runtime = timer.to_model()

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action="scripts/syntheseus/2-run-synth-retro0-local-retro.py",
        sources=[bench_path, stock_path],
        root_dir=BASE_DIR / "data",
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
