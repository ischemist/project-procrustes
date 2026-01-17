"""
Run Synplanner NMCS retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Synplanner's Nested Monte Carlo Search
algorithm and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --directory scripts/planning/run-synplanner 4-run-synp-nmcs.py --benchmark uspto-190
    uv run --directory scripts/planning/run-synplanner 4-run-synp-nmcs.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/synplanner-nmcs[-{effort}]/{benchmark_name}/
"""

import yaml
from synplan.mcts.tree import TreeConfig
from synplan.utils.config import RolloutEvaluationConfig
from synplan.utils.loading import load_evaluation_function, load_reaction_rules
from utils import (
    create_benchmark_parser,
    get_synplanner_paths,
    load_benchmark_and_stock,
    load_policy_from_config,
    run_synplanner_predictions,
    save_synplanner_results,
)

from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()

if __name__ == "__main__":
    parser = create_benchmark_parser("Run Synplanner NMCS (Nested Monte Carlo Search)")
    args = parser.parse_args()

    paths = get_synplanner_paths()
    benchmark, building_blocks, bench_path, stock_path = load_benchmark_and_stock(args.benchmark, paths)

    # Setup output directory
    folder_name = "synplanner-nmcs" if args.effort == "normal" else f"synplanner-nmcs-{args.effort}"
    save_dir = paths.raw_dir / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort}")

    # Load configuration
    config_path = paths.synplanner_dir / "nmcs-config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.effort == "high":
        config["tree"]["max_time"] = 120

    tree_config = TreeConfig.from_dict(config["tree"])

    policy_function = load_policy_from_config(
        policy_params=config.get("node_expansion", {}),
        filtering_weights_path=str(paths.filtering_weights),
        ranking_weights_path=str(paths.ranking_weights),
    )

    # Load resources
    reaction_rules = load_reaction_rules(paths.reaction_rules)

    # Create evaluation function for NMCS
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_function,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=False,
    )
    evaluation_function = load_evaluation_function(eval_config)

    # Run predictions
    logger.info("Retrosynthesis starting with NMCS algorithm")
    results, solved_count, runtime = run_synplanner_predictions(
        benchmark=benchmark,
        tree_config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_function,
        evaluation_function=evaluation_function,
    )

    # Save results
    save_synplanner_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        stock_path=stock_path,
        config_path=config_path,
        script_name="scripts/planning/run-synplanner/4-run-synp-nmcs.py",
        benchmark=benchmark,
    )
