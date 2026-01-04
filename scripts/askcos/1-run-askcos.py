"""
Run ASKCOS retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark by calling the ASKCOS API
and saves results in a structured format matching other prediction scripts.

ASKCOS must be deployed and running on a server. This script makes HTTP calls
to the ASKCOS API endpoint.

Example usage:
    uv run scripts/askcos/1-run-askcos.py --benchmark uspto-190

    uv run scripts/askcos/1-run-askcos.py --benchmark mkt-lin-500 --askcos-url http://localhost:9321/get_buyable_paths

The benchmark definition should be located at: data/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/2-raw/askcos/{benchmark_name}/
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.utils import ExecutionTimer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def call_askcos_api(smiles: str, askcos_url: str, timeout: int = 300) -> dict[str, Any] | None:
    """
    Call the ASKCOS API to get buyable paths for a given SMILES.

    Args:
        smiles: Target molecule SMILES string
        askcos_url: URL of the ASKCOS API endpoint
        timeout: Request timeout in seconds (default: 300)

    Returns:
        JSON response from ASKCOS API, or None if the request fails
    """
    try:
        response = requests.post(
            askcos_url,
            headers={"Content-Type": "application/json"},
            json={"smiles": smiles},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out for SMILES: {smiles}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for SMILES {smiles}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling ASKCOS API for {smiles}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASKCOS retrosynthesis predictions on benchmark targets")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g. uspto-190, mkt-lin-500)",
    )
    parser.add_argument(
        "--askcos-url",
        type=str,
        default="http://0.0.0.0:9321/get_buyable_paths",
        help="URL of the ASKCOS API endpoint (default: http://0.0.0.0:9321/get_buyable_paths)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # Load benchmark
    bench_path = BASE_DIR / "data" / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    logger.info(f"Loading benchmark from {bench_path}")
    benchmark = load_benchmark(bench_path)
    logger.info(f"Loaded {len(benchmark.targets)} targets from benchmark '{benchmark.name}'")

    # Create output directory
    save_dir = BASE_DIR / "data" / "2-raw" / "askcos" / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {save_dir}")

    # Initialize results storage
    results: dict[str, dict[str, Any] | None] = {}
    timer = ExecutionTimer()
    success_count = 0
    failure_count = 0

    logger.info(f"Starting ASKCOS retrosynthesis with endpoint: {args.askcos_url}")
    logger.info(f"Timeout set to {args.timeout} seconds per target")

    # Process each target
    for target in tqdm(benchmark.targets.values(), desc="Processing targets"):
        with timer.measure(target.id):
            try:
                # Call ASKCOS API
                result = call_askcos_api(target.smiles, args.askcos_url, timeout=args.timeout)
                results[target.id] = result

                if result is not None:
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                results[target.id] = None
                failure_count += 1

    runtime = timer.to_model()

    # Save results
    logger.info("Saving results...")
    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")

    # Create manifest
    manifest = create_manifest(
        action="scripts/askcos/1-run-askcos.py",
        sources=[bench_path],
        root_dir=BASE_DIR / "data",
        outputs=[
            (save_dir / "results.json.gz", results, "unknown"),
        ],
        statistics={
            "total_targets": len(benchmark.targets),
            "success_count": success_count,
            "failure_count": failure_count,
            "askcos_url": args.askcos_url,
            "timeout": args.timeout,
        },
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("ASKCOS Processing Complete")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total targets: {len(benchmark.targets)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info(f"{'=' * 60}")
