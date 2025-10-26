"""
Example usage:
 DIRECTMULTISTEP_LOG_LEVEL=WARNING uv run --extra dms --extra torch-gpu scripts/DirectMultiStep/2-run-dms.py --model-name "explorer XL" --use_fp16 --target-name "ursa-bridge-100"
 DIRECTMULTISTEP_LOG_LEVEL=WARNING uv run --extra dms --extra torch-gpu scripts/DirectMultiStep/2-run-dms.py --model-name "flex-20M" --use_fp16 --target-name "ursa-bridge-100"
 DIRECTMULTISTEP_LOG_LEVEL=WARNING uv run --extra dms --extra torch-gpu scripts/DirectMultiStep/2-run-dms.py --model-name "flash" --use_fp16 --target-name "ursa-bridge-100"
 DIRECTMULTISTEP_LOG_LEVEL=WARNING uv run --extra dms --extra torch-gpu scripts/DirectMultiStep/2-run-dms.py --model-name "wide" --use_fp16 --target-name "ursa-bridge-100"

 uv run --extra dms scripts/DirectMultiStep/2-run-dms.py --model-name "flash" --use_fp16 --target-name "test-targets" --device "cpu"
"""

import argparse
import json
import time
from pathlib import Path

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

from ursa.io import load_targets_csv, save_json_gz

base_dir = Path(__file__).resolve().parents[2]


dms_dir = base_dir / "data" / "models" / "dms"
stocks_dir = base_dir / "data" / "models" / "assets"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--ckpt-path", type=Path, help="path to the checkpoint file (if not using a published model)")
    parser.add_argument("--use_fp16", action="store_true", help="Whether to use FP16")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for model inference")
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target")
    desired_device = parser.parse_args().device
    args = parser.parse_args()

    targets = load_targets_csv(base_dir / "data" / "targets" / f"{args.target_name}.csv")

    logger.info(f"model_name: {args.model_name}")
    logger.info(f"use_fp16: {args.use_fp16}")

    logger.info("Loading targets and stock compounds")

    with open(stocks_dir / "ursa-bb-stock-v3-canon.csv") as f:
        ursa_bb_stock_set = set(f.read().splitlines())

    model_name = args.model_name.replace("_", "-").replace(" ", "-")
    folder_name = f"dms-{model_name}-fp16" if args.use_fp16 else f"dms-{model_name}"
    save_dir = base_dir / "data" / "evaluations" / folder_name / args.target_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Retrosythesis starting")
    start = time.time()

    valid_results = {}
    ursa_bb_results = {}
    raw_solved_count = 0
    buyable_solved_count = 0
    emol_solved_count = 0
    ursa_bb_solved_count = 0

    device = ModelFactory.determine_device(desired_device)
    rds = RoutesProcessing(metadata_path=dms_dir / "dms_dictionary.yaml")
    model = load_published_model(args.model_name, dms_dir / "checkpoints", args.use_fp16, force_device=desired_device)

    beam_obj = create_beam_search(model, 50, rds)

    pbar = tqdm(targets.items(), desc="Finding retrosynthetic paths")

    for target_key, target_smiles in pbar:
        target = canonicalize_smiles(target_smiles)

        # this holds all beam search outputs for a SINGLE target, across multiple step calls
        all_beam_results_for_target_NS2: list[list[tuple[str, float]]] = []

        if args.model_name == "explorer XL" or args.model_name == "explorer":
            encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                target, None, None, rds, rds.product_max_length, rds.sm_max_length, args.use_fp16
            )
            beam_result_bs2 = beam_obj.decode(
                src_BC=encoder_inp.to(device),
                steps_B1=steps_tens.to(device) if steps_tens is not None else None,
                path_start_BL=path_tens.to(device),
                progress_bar=False,
            )  # list[list[tuple[str, float]]]
            all_beam_results_for_target_NS2.extend(beam_result_bs2)
        else:
            for step in range(1, 15):
                encoder_inp, steps_tens, path_tens = prepare_input_tensors(
                    target, step, None, rds, rds.product_max_length, rds.sm_max_length, args.use_fp16
                )
                beam_result_bs2 = beam_obj.decode(
                    src_BC=encoder_inp.to(device),
                    steps_B1=steps_tens.to(device) if steps_tens is not None else None,
                    path_start_BL=path_tens.to(device),
                    progress_bar=True,
                )  #  list[list[tuple[str, float]]]

                all_beam_results_for_target_NS2.extend(beam_result_bs2)

        valid_paths_per_batch = find_valid_paths(all_beam_results_for_target_NS2)

        # flatten the list of path-lists into one big list of paths for this target (this is really necessary because we run the model several times with step range(2,9))
        all_valid_paths_for_target = [path for batch in valid_paths_per_batch for path in batch]

        # the processing function expects a list of batches. wrap our flat list to look like a single batch.
        canon_paths_NS2n = canonicalize_paths([all_valid_paths_for_target])
        unique_paths_NS2n = remove_repetitions_within_beam_result(canon_paths_NS2n)

        # unwrap the single batch from the result
        raw_paths = [beam_result[0] for beam_result in unique_paths_NS2n[0]]

        ursa_bb_paths = find_path_strings_with_commercial_sm(raw_paths, commercial_stock=ursa_bb_stock_set)

        raw_solved_count += bool(raw_paths)
        ursa_bb_solved_count += bool(ursa_bb_paths)

        valid_results[target_key] = [eval(p) for p in raw_paths]
        ursa_bb_results[target_key] = [eval(p) for p in ursa_bb_paths]

        # Update progress bar with current path counts
        pbar.set_postfix(
            {
                "Raw:": raw_solved_count,
                "Ursa BB:": ursa_bb_solved_count,
            }
        )

    end = time.time()

    results = {
        "raw_solved_count": raw_solved_count,
        "ursa_bb_solved_count": ursa_bb_solved_count,
        "time_elapsed": end - start,
    }
    logger.info(f"Results: {results}")
    with open(save_dir / "summary.json", "w") as f:
        json.dump(results, f)
    save_json_gz(valid_results, save_dir / "valid_results.json.gz")
    save_json_gz(ursa_bb_results, save_dir / "ursa_bb_results.json.gz")

    usage = """
    python scripts/dms/run-dms-predictions.py --model-name "wide" --use_fp16 --target-name "ursa-bridge-100" --device "cuda:0"
    """
    logger.info(usage)
