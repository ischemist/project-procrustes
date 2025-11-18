"""
Scores processed predictions against a benchmark and stock.

Calculates solvability and ranks ground truth matches.
This step generates the intermediate data needed for statistical analysis.

Usage:
    uv run scripts/00-score.py --model dms-flash-fp16 --benchmark random-n5-100
"""

import argparse
from pathlib import Path

from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.io.routes import load_routes
from retrocast.utils.logging import logger
from retrocast.workflow.score import score_model

# CONFIG
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main():
    parser = argparse.ArgumentParser(description="Score model predictions.")
    parser.add_argument("--model", required=True, help="Name of the model (e.g. dms-flash-fp16)")
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark (e.g. stratified-linear-600)")
    # Optional: Allow overriding stock, but default to what the benchmark says
    parser.add_argument("--stock", help="Name of the stock file (defaults to benchmark's required stock)")

    args = parser.parse_args()

    # 1. Resolve Paths
    # Input: Benchmark Definition
    bench_path = DATA_DIR / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    if not bench_path.exists():
        logger.error(f"Benchmark not found: {bench_path}")
        return

    # Input: Processed Routes
    # data/processed/{benchmark}/{model}/routes.json.gz
    routes_path = DATA_DIR / "3-processed" / args.benchmark / args.model / "routes.json.gz"
    if not routes_path.exists():
        logger.error(f"Predictions not found: {routes_path}")
        logger.error("Did you run the ingestion step first?")
        return

    # Load Benchmark
    benchmark = load_benchmark(bench_path)

    # Determine Stock
    stock_name = args.stock or benchmark.stock_name
    if not stock_name:
        logger.error("No stock specified in benchmark or arguments.")
        return

    # Input: Stock File
    stock_path = DATA_DIR / "1-benchmarks" / "stocks" / f"{stock_name}.txt"
    if not stock_path.exists():
        logger.error(f"Stock file not found: {stock_path}")
        return

    # 2. Load Data
    predictions = load_routes(routes_path)

    # 3. Run Scoring Workflow
    # This is where the heavy lifting happens (solvability check, etc)
    eval_results = score_model(
        benchmark=benchmark, predictions=predictions, stock_path=stock_path, model_name=args.model
    )

    # 4. Save Output
    # data/scored/{benchmark}/{model}/{stock}/evaluation.json.gz
    output_dir = DATA_DIR / "4-scored" / args.benchmark / args.model / stock_name
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "evaluation.json.gz"
    save_json_gz(eval_results, out_path)

    # 5. Create Manifest
    # We track inputs (benchmark, stock, predictions) and output (evaluation)
    manifest = create_manifest(
        action="score_model",
        sources=[bench_path, routes_path, stock_path],
        outputs=[(out_path, eval_results)],
        parameters={"model": args.model, "benchmark": args.benchmark, "stock": stock_name},
        statistics={
            "n_targets": len(eval_results.results),
            "n_solvable": sum(1 for r in eval_results.results.values() if r.is_solvable),
        },
    )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Scoring complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
