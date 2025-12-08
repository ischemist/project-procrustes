"""
Run Synplanner MCTS retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Synplanner's MCTS algorithm
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --extra synplanner scripts/synplanner/2-run-synp-eval.py --benchmark uspto-190
    uv run --extra synplanner scripts/synplanner/2-run-synp-eval.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/synplanner-{stock}[-{effort}]/{benchmark_name}/
"""

import argparse
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

from retrocast.io import create_manifest, load_benchmark, load_stock_file, save_execution_stats, save_json_gz
from retrocast.models.benchmark import ExecutionStats
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]

SYNPLANNER_DIR = BASE_DIR / "data" / "0-assets" / "model-configs" / "synplanner"
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

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {args.benchmark}"

    # 2. Load Stock
    stock_path = STOCKS_DIR / f"{benchmark.stock_name}.csv.gz"
    building_blocks = load_stock_file(stock_path)

    # 3. Setup Output
    folder_name = "synplanner-eval" if args.effort == "normal" else f"synplanner-{args.effort}"
    save_dir = BASE_DIR / "data" / "2-raw" / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort}")

    # 4. Load Model Configuration
    config_filename = "search-eval-config-high.yaml" if args.effort == "high" else "search-eval-config.yaml"
    logger.info(f"using config: {config_filename}")
    config_path = SYNPLANNER_DIR / config_filename
    value_network_path = SYNPLANNER_DIR / "uspto" / "weights" / "value_network.ckpt"
    rank_weights = SYNPLANNER_DIR / "uspto" / "weights" / "ranking_policy_network.ckpt"
    reaction_rules_path = SYNPLANNER_DIR / "uspto" / "uspto_reaction_rules.pickle"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    search_config = {**config["tree"], **config["node_evaluation"]}
    policy_config = PolicyNetworkConfig.from_dict({**config["node_expansion"], **{"weights_path": rank_weights}})

    # 5. Load Resources
    policy_function = PolicyNetworkFunction(policy_config=policy_config)
    reaction_rules = load_reaction_rules(reaction_rules_path)

    if search_config["evaluation_type"] == "gcn" and value_network_path.exists():
        value_function = ValueNetworkFunction(weights_path=str(value_network_path))
    else:
        value_function = None

    tree_config = TreeConfig.from_dict(search_config)

    # 6. Run Predictions
    logger.info("Retrosynthesis starting")

    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    runtime = ExecutionStats()

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()

        try:
            target_mol = mol_from_smiles(target.smiles)
            if not target_mol:
                logger.warning(f"Could not create molecule for target {target.id} ({target.smiles}). Skipping.")
                results[target.id] = []
            else:
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
                    results[target.id] = list(raw_routes.values())
                    solved_count += 1
                else:
                    results[target.id] = []

        except Exception as e:
            logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
            results[target.id] = []
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
        action="scripts/synplanner/2-run-synp-eval.py",
        sources=[bench_path, stock_path],
        root_dir=BASE_DIR / "data",
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")
