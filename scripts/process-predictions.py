"""
unified pipeline runner for ursa.

this script uses `ursa-config.yaml` to orchestrate pre-processing and processing
of model outputs for the benchmark.

---
usage examples:
---

# show available models defined in the config
uv run scripts/process-predictions.py list

# show pre-processing steps for a specific model
uv run scripts/process-predictions.py info --model multistep-ttl

# process the output of a model for a given dataset
uv run scripts/process-predictions.py process --model dms-flash-fp16 --dataset ursa-bridge-100

# process output, but override the sampling strategy from the config
uv run scripts/process-predictions.py process --model dms-flash-fp16 --dataset uspto-190 --sampling-strategy top-k --k 5

# process all models for a specific dataset
uv run scripts/process-predictions.py process --dataset rs-first-25 --all-models

# process a specific model for all known datasets
uv run scripts/process-predictions.py process --model aizynthfinder-mcts --all-datasets
"""

import argparse
from pathlib import Path

from retrocast import cli


def main() -> None:
    """main function to parse arguments and delegate to handlers in `ursa.cli`."""
    parser = argparse.ArgumentParser(description="ursa pipeline runner.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, help="available commands")

    # --- list command ---
    subparsers.add_parser("list", help="list all models defined in `ursa-config.yaml`.")

    # --- info command ---
    parser_info = subparsers.add_parser("info", help="show configuration details for a specific model.")
    parser_info.add_argument("--model", required=True, type=str, help="the name of the model.")

    # --- process command ---
    parser_process = subparsers.add_parser("process", help="process raw model outputs into the benchmark format.")
    group = parser_process.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="the name of the model to process.")
    group.add_argument("--all-models", action="store_true", help="process all models defined in the config.")

    group2 = parser_process.add_mutually_exclusive_group(required=True)
    group2.add_argument("--dataset", type=str, help="the name of the dataset to process.")
    group2.add_argument("--all-datasets", action="store_true", help="process for all available datasets.")

    parser_process.add_argument(
        "--sampling-strategy", type=str, default=None, help="override the default sampling strategy from the config."
    )
    parser_process.add_argument("--k", type=int, default=None, help="override the default k value from the config.")

    args = parser.parse_args()

    # the base directory is the parent of the 'scripts' directory.
    base_dir = Path(__file__).resolve().parents[1]
    config = cli.load_config(base_dir / "ursa-config.yaml")

    if args.command == "list":
        cli.handle_list(config)
    elif args.command == "info":
        cli.handle_info(config, args.model)
    elif args.command == "process":
        cli.handle_process(base_dir, config, args)


if __name__ == "__main__":
    main()
