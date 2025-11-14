import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from retrocast.adapters.factory import get_adapter
from retrocast.core import process_model_run
from retrocast.exceptions import RetroCastException
from retrocast.io import load_and_prepare_targets
from retrocast.utils.logging import logger


def load_config(config_path: Path) -> dict[str, Any]:
    """loads the main yaml configuration file."""
    try:
        with config_path.open("r") as f:
            return cast(dict[str, Any], yaml.safe_load(f))
    except FileNotFoundError:
        logger.error(f"configuration file not found at {config_path}. please create `ursa-config.yaml`.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"error parsing configuration file {config_path}: {e}")
        sys.exit(1)


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """retrieves the configuration for a specific model."""
    if "models" not in config or model_name not in config["models"]:
        logger.error(f"model '{model_name}' not found in `ursa-config.yaml`.")
        sys.exit(1)
    return cast(dict[str, Any], config["models"][model_name])


def handle_list(config: dict[str, Any]) -> None:
    """handles the 'list' command."""
    logger.info("available models in `ursa-config.yaml`:")
    for model_name in sorted(config.get("models", {}).keys()):
        print(f"  - {model_name}")


def handle_info(config: dict[str, Any], model_name: str) -> None:
    """handles the 'info' command."""
    model_conf = get_model_config(config, model_name)
    logger.info(f"configuration for model: '{model_name}'")

    preprocess_script = model_conf.get("preprocess_script")
    if preprocess_script:
        logger.info("  - pre-processing required. run this script first:")
        print(f"    uv run {preprocess_script}")
    else:
        logger.info("  - no pre-processing script defined.")

    raw_file = model_conf.get("raw_results_filename", "not specified")
    logger.info(f"  - expected raw results filename: {raw_file}")

    sampling = model_conf.get("sampling")
    if sampling:
        strategy = sampling.get("strategy")
        k = sampling.get("k")
        logger.info(f"  - default sampling: strategy='{strategy}', k={k}")
    else:
        logger.info("  - default sampling: none")


def _run_single_process(
    base_dir: Path,
    config: dict[str, Any],
    model_name: str,
    dataset_name: str,
    sampling_strategy_override: str | None,
    k_override: int | None,
) -> None:
    """runs the processing for a single model and dataset combination."""
    logger.info(f"--- processing model '{model_name}' on dataset '{dataset_name}' ---")
    model_conf = get_model_config(config, model_name)

    # determine paths
    raw_file = base_dir / "data" / "evaluations" / model_name / dataset_name / model_conf["raw_results_filename"]
    output_dir = base_dir / "data" / "processed" / dataset_name
    targets_file = base_dir / "data" / "targets" / f"{dataset_name}.csv"

    if not raw_file.exists():
        logger.error(f"raw results file not found: {raw_file}")
        if model_conf.get("preprocess_script"):
            logger.error(f"did you run the pre-processing script for '{model_name}'?")
        return  # continue to next run instead of exiting

    if not targets_file.exists():
        logger.error(f"targets file not found: {targets_file}")
        return

    # determine sampling strategy
    sampling_conf = model_conf.get("sampling")
    strategy = sampling_strategy_override
    k = k_override
    if strategy is None and sampling_conf:
        strategy = sampling_conf.get("strategy")
    if k is None and sampling_conf:
        k = sampling_conf.get("k")

    try:
        targets_map = load_and_prepare_targets(targets_file)
        adapter = get_adapter(model_conf["adapter"])

        process_model_run(
            model_name=model_name,
            adapter=adapter,
            dataset_name=dataset_name,
            raw_results_file=raw_file,
            processed_dir=output_dir,
            targets_map=targets_map,
            sampling_strategy=strategy,
            sample_k=k,
        )
    except RetroCastException as e:
        logger.error(f"a critical error occurred during processing for {model_name}/{dataset_name}: {e}")
    except Exception as e:
        logger.critical(f"an unexpected error occurred for {model_name}/{dataset_name}: {e}", exc_info=True)


def handle_process(
    base_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """handles the 'process' command, orchestrating single or multiple runs."""
    all_models = sorted(config.get("models", {}).keys())
    # this is a bit brittle, but good enough for now.
    all_datasets = sorted(
        [p.stem for p in (base_dir / "data" / "targets").glob("*.csv") if p.is_file() and p.name != "README.md"]
    )

    models_to_run = all_models if args.all_models else [args.model]
    datasets_to_run = all_datasets if args.all_datasets else [args.dataset]

    if not any(m for m in models_to_run if m) or not any(d for d in datasets_to_run if d):
        logger.error("you must specify a model and dataset, or use --all-models/--all-datasets.")
        sys.exit(1)

    for model in models_to_run:
        for dataset in datasets_to_run:
            _run_single_process(base_dir, config, model, dataset, args.sampling_strategy, args.k)
            logger.info("-" * 80)
