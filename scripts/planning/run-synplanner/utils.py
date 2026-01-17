"""Shared utilities for Synplanner scripts."""

from __future__ import annotations

import argparse
import gzip
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synplan.chem.reaction_routes.io import make_json
from synplan.chem.reaction_routes.route_cgr import extract_reactions
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree, TreeConfig
from synplan.utils.config import CombinedPolicyConfig, PolicyNetworkConfig
from synplan.utils.loading import load_building_blocks, load_combined_policy_function, load_policy_function
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.paths import get_paths
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger


@dataclass
class SynplannerPaths:
    """Standard paths for Synplanner resources."""

    synplanner_dir: Path
    stocks_dir: Path
    benchmarks_dir: Path
    raw_dir: Path
    filtering_weights: Path
    ranking_weights: Path
    reaction_rules: Path


def get_synplanner_paths() -> SynplannerPaths:
    """Get standard Synplanner paths using project root resolution.

    Computes the project root from this file's location to ensure paths
    resolve correctly regardless of working directory (needed when running
    with `uv run --directory` to use the local lockfile).

    Returns:
        SynplannerPaths with all standard resource paths.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "retrocast"
    paths = get_paths(data_dir)
    synplanner_dir = data_dir / "0-assets" / "model-configs" / "synplanner"

    return SynplannerPaths(
        synplanner_dir=synplanner_dir,
        stocks_dir=paths["stocks"],
        benchmarks_dir=paths["benchmarks"],
        raw_dir=paths["raw"],
        filtering_weights=synplanner_dir / "uspto" / "weights" / "filtering_policy_network.ckpt",
        ranking_weights=synplanner_dir / "uspto" / "weights" / "ranking_policy_network.ckpt",
        reaction_rules=synplanner_dir / "uspto" / "uspto_reaction_rules.pickle",
    )


def create_benchmark_parser(description: str) -> argparse.ArgumentParser:
    """Create standard argument parser for Synplanner benchmark scripts.

    Args:
        description: Script description for help text.

    Returns:
        Configured ArgumentParser with --benchmark and --effort arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g. stratified-linear-600)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="normal",
        choices=["normal", "high"],
        help="Search effort level: normal or high",
    )
    return parser


def load_benchmark_and_stock(
    benchmark_name: str,
    paths: SynplannerPaths,
) -> tuple[BenchmarkSet, set[str], Path, Path]:
    """Load benchmark definition and corresponding building blocks.

    Args:
        benchmark_name: Name of the benchmark (without extension).
        paths: SynplannerPaths instance with resource locations.

    Returns:
        Tuple of (benchmark, building_blocks, bench_path, stock_path).

    Raises:
        AssertionError: If benchmark has no stock_name defined.
    """
    bench_path = paths.benchmarks_dir / f"{benchmark_name}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {benchmark_name}"

    stock_path = paths.stocks_dir / f"{benchmark.stock_name}.csv.gz"
    building_blocks = load_building_blocks_cached(stock_path)

    return benchmark, building_blocks, bench_path, stock_path


def load_policy_from_config(
    policy_params: dict,
    filtering_weights_path: str,
    ranking_weights_path: str,
) -> Callable:
    """Loads the appropriate policy function based on configuration.

    Args:
        policy_params: Dictionary containing policy configuration, including 'mode'
            ('ranking' or 'combined'), 'top_rules', and 'rule_prob_threshold'.
        filtering_weights_path: Path to the filtering policy network weights.
        ranking_weights_path: Path to the ranking policy network weights.

    Returns:
        The loaded policy function callable.
    """
    mode = policy_params.get("mode", "ranking")
    if mode == "combined":
        combined_policy_config = CombinedPolicyConfig(
            filtering_weights_path=filtering_weights_path,
            ranking_weights_path=ranking_weights_path,
            top_rules=policy_params.get("top_rules", 50),
            rule_prob_threshold=policy_params.get("rule_prob_threshold", 0.0),
        )
        return load_combined_policy_function(combined_config=combined_policy_config)
    # 'ranking' or other modes
    return load_policy_function(policy_config=PolicyNetworkConfig(weights_path=ranking_weights_path))


def run_synplanner_predictions(
    benchmark: BenchmarkSet,
    tree_config: TreeConfig,
    reaction_rules: Any,
    building_blocks: set[str],
    expansion_function: Callable,
    evaluation_function: Callable,
) -> tuple[dict[str, list[dict[str, Any]]], int, ExecutionStats]:
    """Run Synplanner search over all benchmark targets.

    Args:
        benchmark: Benchmark containing targets to process.
        tree_config: Configuration for the search tree.
        reaction_rules: Loaded reaction rules.
        building_blocks: Set of building block SMILES.
        expansion_function: Policy function for node expansion.
        evaluation_function: Evaluation function for node scoring.

    Returns:
        Tuple of (results_dict, solved_count, execution_runtime).
    """
    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                target_mol = mol_from_smiles(target.smiles, standardize=True)
                if not target_mol:
                    logger.warning(f"Could not create molecule for target {target.id} ({target.smiles}). Skipping.")
                    results[target.id] = []
                else:
                    search_tree = Tree(
                        target=target_mol,
                        config=tree_config,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        expansion_function=expansion_function,
                        evaluation_function=evaluation_function,
                    )

                    # run the search
                    _ = list(search_tree)

                    if bool(search_tree.winning_nodes):
                        raw_routes = make_json(extract_reactions(search_tree))
                        results[target.id] = list(raw_routes.values())
                        solved_count += 1
                    else:
                        results[target.id] = []

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                results[target.id] = []

    return results, solved_count, timer.to_model()


def save_synplanner_results(
    results: dict[str, list[dict[str, Any]]],
    runtime: ExecutionStats,
    save_dir: Path,
    bench_path: Path,
    stock_path: Path,
    config_path: Path,
    script_name: str,
    benchmark: BenchmarkSet,
) -> None:
    """Save Synplanner results, execution stats, and manifest.

    Args:
        results: Dictionary mapping target IDs to route lists.
        runtime: Execution timing information.
        save_dir: Directory to save outputs.
        bench_path: Path to benchmark definition file.
        stock_path: Path to stock file.
        config_path: Path to config file used.
        script_name: Name of the calling script (for manifest).
        benchmark: Benchmark object (for statistics).
    """
    solved_count = sum(1 for routes in results.values() if routes)

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")

    manifest = create_manifest(
        action=script_name,
        sources=[bench_path, stock_path, config_path],
        root_dir=save_dir.parents[2],  # data/ directory
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")


def load_building_blocks_cached(
    stock_path: Path,
    *,
    silent: bool = False,
) -> set[str]:
    """Load building blocks with caching for SynPlanner's standardization.

    SynPlanner uses special canonicalization that takes ~5 minutes for large stocks.
    This function checks for a pre-standardized cache file and uses it if available,
    otherwise standardizes and saves the result for future runs.

    Args:
        stock_path: Path to the original stock CSV file (e.g., buyables-stock.csv.gz).
        silent: Suppress progress output from load_building_blocks.

    Returns:
        Set of SMILES strings representing building blocks.
    """
    # Check for cached standardized version (e.g., buyables-stock-synplanner.csv.gz)
    cached_path = stock_path.with_name(stock_path.name.replace(".csv.gz", "-synplanner.csv.gz"))

    if cached_path.exists():
        return load_building_blocks(cached_path, standardize=False, silent=silent)

    # Load with standardization (slow ~5 min)
    building_blocks = load_building_blocks(stock_path, standardize=True, silent=silent)

    # Save cached version for next time
    with gzip.open(cached_path, "wt", encoding="utf-8") as f:
        f.write("SMILES\n")  # header
        for smiles in building_blocks:
            f.write(f"{smiles}\n")

    return building_blocks
