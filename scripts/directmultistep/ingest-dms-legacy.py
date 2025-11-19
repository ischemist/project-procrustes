"""
Ingests legacy DMS predictions (pickle format) into the retrocast processed format.

Usage:
    uv run scripts/directmultistep/ingest-dms-legacy.py --benchmark stratified-convergent-250
    uv run scripts/directmultistep/ingest-dms-legacy.py --benchmark stratified-linear-600
    uv run scripts/directmultistep/ingest-dms-legacy.py --benchmark random-n5-500
"""

import argparse
import pickle
from pathlib import Path

from retrocast import adapt_routes
from retrocast.domain.chem import canonicalize_smiles
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.io.routes import save_routes
from retrocast.models.chem import Route
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
ds = "n5"
PICKLE_PATH = BASE_DIR / "data" / "2-raw" / "dms-flash-fp16" / ds / f"{ds}_correct_paths_NS2n.pkl"


def load_legacy_pickle(path: Path) -> list[list[tuple[str, float]]]:
    """Loads the raw pickle list-of-lists."""
    logger.info(f"Loading pickle {path}...")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser("Ingest DMS Legacy Data")
    parser.add_argument("--benchmark", type=str, default="benchmark_name", help="Name of the benchmark")
    args = parser.parse_args()
    # 1. Load the Benchmark Definition
    bench_def_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_def_path)

    # Create lookup map: {SMILES -> [target_id_1, target_id_2]}
    smiles_map = benchmark.get_smiles_map()

    # 2. Load Predictions
    raw_data = load_legacy_pickle(PICKLE_PATH)

    # 2. Initialize with EMPTY lists for ALL targets
    # This ensures we have a denominator for every target in the benchmark
    processed_predictions: dict[str, list[Route]] = {tid: [] for tid in benchmark.targets}

    hits = 0

    # 3. Iterate and Match
    for prediction_group in raw_data:
        if not prediction_group:
            continue

        first_route_str = prediction_group[0][0]
        first_route_dict = eval(first_route_str)
        canon_smiles = canonicalize_smiles(first_route_dict.get("smiles"))

        if canon_smiles in smiles_map:
            target_ids = smiles_map[canon_smiles]
            raw_routes = [eval(p[0]) for p in prediction_group]

            for tid in target_ids:
                target_obj = benchmark.targets[tid]
                adapted_routes = adapt_routes(raw_routes, target_obj, "dms")
                processed_predictions[tid] = adapted_routes
                hits += 1

    logger.info(f"Matched {hits} targets from pickle out of {len(benchmark.targets)} in benchmark.")

    # 4. Save
    model_name = "dms-flash-fp16"
    output_dir = BASE_DIR / "data" / "3-processed" / args.benchmark / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "routes.json.gz"
    save_routes(processed_predictions, out_path)

    # 5. Manifest
    manifest = create_manifest(
        action="scripts/directmultistep/ingest-dms-legacy",
        sources=[PICKLE_PATH],
        outputs=[(out_path, processed_predictions)],
        parameters={"benchmark": args.benchmark, "model": model_name},
        statistics={"n_targets_found": hits},
    )

    with open(output_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
