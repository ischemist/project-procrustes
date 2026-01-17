"""
Run Synplanner MCTS retrosynthesis predictions on a batch of targets using value-network evaluation.

This script processes targets from a benchmark using Synplanner's MCTS algorithm
with evaluation-first search guided by a value network and saves results in a structured
format matching other prediction scripts.

Example usage:
    uv run --directory scripts/planning/run-synplanner 2-run-synp-val.py --benchmark mkt-cnv-160
    uv run --directory scripts/planning/run-synplanner 2-run-synp-val.py --benchmark random-n5-2-seed=20251030 --effort high

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/synplanner-{stock}[-{effort}]/{benchmark_name}/
"""

import yaml
from synplan.mcts.tree import TreeConfig
from synplan.utils.config import ValueNetworkEvaluationConfig
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
    parser = create_benchmark_parser("Run Synplanner MCTS with value-network evaluation")
    args = parser.parse_args()

    paths = get_synplanner_paths()
    benchmark, building_blocks, bench_path, stock_path = load_benchmark_and_stock(args.benchmark, paths)

    # Setup output directory
    folder_name = "synplanner-mcts-val" if args.effort == "normal" else f"synplanner-mcts-val-{args.effort}"
    save_dir = paths.raw_dir / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort}")

    # Load configuration
    config_path = paths.synplanner_dir / "mcts-val-config.yaml"
    value_network_path = paths.synplanner_dir / "uspto" / "weights" / "value_network.ckpt"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.effort == "high":
        config["tree"]["max_iterations"] = 500

    tree_config = TreeConfig.from_dict(config["tree"])
    tree_config.search_strategy = "evaluation_first"
    tree_config.evaluation_agg = config["node_evaluation"].get("evaluation_agg", tree_config.evaluation_agg)

    policy_function = load_policy_from_config(
        policy_params=config.get("node_expansion", {}),
        filtering_weights_path=str(paths.filtering_weights),
        ranking_weights_path=str(paths.ranking_weights),
    )

    # Load resources
    reaction_rules = load_reaction_rules(paths.reaction_rules)

    evaluation_type = str(config["node_evaluation"].get("evaluation_type", "")).lower()
    if evaluation_type and evaluation_type != "gcn":
        logger.warning(f"Config evaluation_type={evaluation_type!r} ignored; using value network evaluation.")

    if not value_network_path.exists():
        raise FileNotFoundError(f"Value network weights not found at {value_network_path}")

    eval_config = ValueNetworkEvaluationConfig(weights_path=str(value_network_path))
    evaluation_function = load_evaluation_function(eval_config)

    # Run predictions
    logger.info("Retrosynthesis starting")
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
        script_name="scripts/planning/run-synplanner/2-run-synp-val.py",
        benchmark=benchmark,
    )
