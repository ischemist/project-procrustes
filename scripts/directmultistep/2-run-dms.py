"""
Run DirectMultiStep (DMS) retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using DirectMultiStep algorithm
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --extra dms scripts/directmultistep/2-run-dms.py --benchmark mkt-cnv-160 --model-name "explorer XL" --device cuda --use_fp16
    uv run --extra dms scripts/directmultistep/2-run-dms.py --benchmark ref-cnv-400 --model-name "explorer XL" --device cuda --use_fp16
    uv run --extra dms scripts/directmultistep/2-run-dms.py --benchmark uspto-190 --model-name "explorer XL" --device cuda --use_fp16

    uv run --extra dms scripts/directmultistep/2-run-dms.py --benchmark uspto-190 --model-name "flash" --device cuda --use_fp16

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/dms-{model_name}/{benchmark_name}/
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from directmultistep.generate import create_beam_search, load_published_model, prepare_input_tensors
from directmultistep.model import ModelFactory
from directmultistep.utils.dataset import RoutesProcessing
from directmultistep.utils.logging_config import logger
from directmultistep.utils.post_process import (
    canonicalize_paths,
    find_path_strings_with_commercial_sm,
    find_valid_paths,
    remove_repetitions_within_beam_result,
)
from directmultistep.utils.pre_process import canonicalize_smiles
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, load_stock_file, save_execution_stats, save_json_gz
from retrocast.utils import ExecutionTimer

logger.setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parents[2]

DMS_DIR = BASE_DIR / "data" / "0-assets" / "model-configs" / "dms"
STOCKS_DIR = BASE_DIR / "data" / "1-benchmarks" / "stocks"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Name of the benchmark set (e.g. stratified-linear-600)"
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model (e.g. flash, wide, explorer XL)"
    )
    parser.add_argument("--ckpt-path", type=Path, help="path to the checkpoint file (if not using a published model)")
    parser.add_argument("--use_fp16", action="store_true", help="Whether to use FP16")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for model inference")
    args = parser.parse_args()

    # 1. Load Benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)

    logger.info(f"model_name: {args.model_name}")
    logger.info(f"use_fp16: {args.use_fp16}")

    logger.info("Loading stock compounds")
    stocks = {
        "n1-n5": load_stock_file(STOCKS_DIR / "n1-n5-stock.csv.gz"),
        "buyables": load_stock_file(STOCKS_DIR / "buyables-stock.csv.gz"),
    }

    model_name = args.model_name.replace("_", "-").replace(" ", "-")
    folder_name = f"dms-{model_name}"
    save_dir = BASE_DIR / "data" / "2-raw" / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Retrosynthesis starting")

    valid_results: dict[str, dict[str, Any]] = {}
    buyables_results: dict[str, dict[str, Any]] = {}
    n1n5_results: dict[str, dict[str, Any]] = {}
    raw_solved_count = 0
    solved_counts = {stock_name: 0 for stock_name in stocks}
    timer = ExecutionTimer()

    device = ModelFactory.determine_device(args.device)
    rds = RoutesProcessing(metadata_path=DMS_DIR / "dms_dictionary.yaml")
    model = load_published_model(args.model_name, DMS_DIR / "checkpoints", args.use_fp16, force_device=args.device)

    beam_obj = create_beam_search(model, 50, rds)

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                target_smiles = canonicalize_smiles(target.smiles)

                # this holds all beam search outputs for a SINGLE target, across multiple step calls
                all_beam_results_for_target_NS2: list[list[tuple[str, float]]] = []

                if args.model_name == "explorer XL" or args.model_name == "explorer":
                    encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                        target_smiles, None, None, rds, rds.product_max_length, rds.sm_max_length, args.use_fp16
                    )
                    beam_result_bs2 = beam_obj.decode(
                        src_BC=encoder_inp.to(device),
                        steps_B1=steps_tens.to(device) if steps_tens is not None else None,
                        path_start_BL=path_tens.to(device),
                        progress_bar=False,
                    )  # list[list[tuple[str, float]]]
                    all_beam_results_for_target_NS2.extend(beam_result_bs2)
                else:
                    for step in range(1, 11):
                        encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                            target_smiles, step, None, rds, rds.product_max_length, rds.sm_max_length, args.use_fp16
                        )
                        beam_result_bs2 = beam_obj.decode(
                            src_BC=encoder_inp.to(device),
                            steps_B1=steps_tens.to(device) if steps_tens is not None else None,
                            path_start_BL=path_tens.to(device),
                            progress_bar=False,
                        )  #  list[list[tuple[str, float]]]

                        all_beam_results_for_target_NS2.extend(beam_result_bs2)

                valid_paths_per_batch = find_valid_paths(all_beam_results_for_target_NS2)

                # flatten the list of path-lists into one big list of paths for this target
                all_valid_paths_for_target = [path for batch in valid_paths_per_batch for path in batch]

                # the processing function expects a list of batches. wrap our flat list to look like a single batch.
                canon_paths_NS2n = canonicalize_paths([all_valid_paths_for_target])
                unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n)

                # unwrap the single batch from the result
                raw_paths = [beam_result[0] for beam_result in unique_paths_NS2n[0]]

                raw_solved_count += bool(raw_paths)

                valid_results[target.id] = [eval(p) for p in raw_paths]
                buyables_paths = find_path_strings_with_commercial_sm(raw_paths, commercial_stock=stocks["buyables"])
                buyables_results[target.id] = [eval(p) for p in buyables_paths]
                solved_counts["buyables"] += bool(buyables_paths)
                n1n5_paths = find_path_strings_with_commercial_sm(raw_paths, commercial_stock=stocks["n1-n5"])
                n1n5_results[target.id] = [eval(p) for p in n1n5_paths]
                solved_counts["n1-n5"] += bool(n1n5_paths)

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                valid_results[target.id] = []
                buyables_results[target.id] = []
                n1n5_results[target.id] = []

    runtime = timer.to_model()

    summary = {
        "raw_solved_count": raw_solved_count,
        "total_targets": len(benchmark.targets),
    }
    summary.update({f"{stock_name}_solved_count": count for stock_name, count in solved_counts.items()})

    save_json_gz(valid_results, save_dir / "valid_results.json.gz")
    save_json_gz(buyables_results, save_dir / "buyables_results.json.gz")
    save_json_gz(n1n5_results, save_dir / "n1n5_results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action="scripts/directmultistep/2-run-dms.py",
        sources=[bench_path],
        root_dir=BASE_DIR / "data",
        outputs=[
            (save_dir / "valid_results.json.gz", valid_results, "unknown"),
            (save_dir / "buyables_results.json.gz", buyables_results, "unknown"),
            (save_dir / "n1n5_results.json.gz", n1n5_results, "unknown"),
        ],
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Raw solved: {raw_solved_count}")
    for stock_name, count in solved_counts.items():
        logger.info(f"{stock_name.capitalize()} solved: {count}")
