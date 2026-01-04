"""
Run Syntheseus LocalRetroModel retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Syntheseus's LocalRetroModel
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --extra syntheseus scripts/syntheseus/1-run-synth-bfs-local-retro.py --benchmark random-n5-2-seed=20251030
    uv run --extra syntheseus scripts/syntheseus/1-run-synth-bfs-local-retro.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/syntheseus-bfs-local-retro[-{effort}]/{benchmark_name}/
"""

import argparse
from pathlib import Path
from typing import Any

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.mol_inventory import SmilesListInventory
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, load_stock_file, save_execution_stats, save_json_gz
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
    stock_path = STOCKS_DIR / f"{benchmark.stock_name}.txt"
    building_blocks = load_stock_file(stock_path)

    # 3. Setup Output
    folder_name = (
        "syntheseus-bfs-local-retro" if args.effort == "normal" else f"syntheseus-bfs-local-retro-{args.effort}"
    )
    save_dir = BASE_DIR / "data" / "2-raw" / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort}")

    # 4. Set up inventory with the building blocks
    inventory = SmilesListInventory(smiles_list=building_blocks)

    # 5. Set up the reaction model
    model = LocalRetroModel(use_cache=True, default_num_results=10)

    # 6. Run Predictions
    logger.info("Retrosynthesis starting")

    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                # Set up search algorithm for each target
                search_algorithm = AndOr_BreadthFirstSearch(
                    reaction_model=model,
                    mol_inventory=inventory,
                    limit_iterations=iterations,  # max number of algorithm iterations
                    limit_reaction_model_calls=100,  # max number of model calls
                    time_limit_s=60.0,  # max runtime in seconds
                )

                # Run search
                test_mol = Molecule(target.smiles)
                output_graph, _ = search_algorithm.run_from_mol(test_mol)

                # Extract routes
                routes = list(iter_routes_time_order(output_graph, max_routes=10))

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
        action="scripts/syntheseus/1-run-synth-bfs-local-retro.py",
        sources=[bench_path, stock_path],
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
