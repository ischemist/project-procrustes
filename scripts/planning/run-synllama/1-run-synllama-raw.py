"""
Run SynLlama retrosynthesis predictions on a batch of targets with raw generation (no BB reconstruction).

This script processes targets from a benchmark using SynLlama's LLM-based synthesis
planning in raw generation mode, which provides direct LLM output without building block
reconstruction. The synthesis strings may contain arbitrary building blocks that are not
necessarily in your stock.

Example usage:
    uv run --directory scripts/planning/run-synllama 1-run-synllama-raw.py --benchmark mkt-lin-500
    uv run --directory scripts/planning/run-synllama 1-run-synllama-raw.py --benchmark random-n5-2-seed=20251030
    uv run --directory scripts/planning/run-synllama 1-run-synllama-raw.py --benchmark mkt-lin-500 --rxn-set 115rxns

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/retrocast/2-raw/synllama-{version}-raw-{rxn_set}/{benchmark_name}/

Note: This mode does NOT require building block indices (no need to run 2-build-bb-indices.py).
      For stock-aware planning with BB reconstruction, use 3-run-synllama-reconstruction.py instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synllama.api import plan_synthesis
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.paths import get_paths
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()

# SynLlama version - update when upgrading the library
PLANNER_VERSION = "0.1.0"


@dataclass
class SynLlamaPaths:
    """Standard paths for SynLlama resources."""

    synllama_data: Path
    model_path: Path
    benchmarks_dir: Path
    raw_dir: Path


def get_synllama_paths(rxn_set: str) -> SynLlamaPaths:
    """Get standard SynLlama paths using project root resolution.

    Args:
        rxn_set: Reaction set to use ("91rxns" or "115rxns").

    Returns:
        SynLlamaPaths with all standard resource paths.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "retrocast"
    paths = get_paths(data_dir)

    # synllama-data is in scripts/planning/run-synllama/
    synllama_data = Path(__file__).parent / "synllama-data"

    model_path = synllama_data / "inference" / "model" / f"SynLlama-1B-2M-{rxn_set}"

    return SynLlamaPaths(
        synllama_data=synllama_data,
        model_path=model_path,
        benchmarks_dir=paths["benchmarks"],
        raw_dir=paths["raw"],
    )


def run_synllama_raw_predictions(
    benchmark: BenchmarkSet,
    model_path: Path,
    max_length: int = 1600,
    device: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int, ExecutionStats]:
    """Run SynLlama raw generation over all benchmark targets.

    Args:
        benchmark: Benchmark containing targets to process.
        model_path: Path to SynLlama model directory.
        max_length: Maximum tokens to generate.
        device: Device to use ("cuda", "cpu", or None for auto-detect).

    Returns:
        Tuple of (results_dict, solved_count, execution_runtime).
    """
    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    for target in tqdm(benchmark.targets.values(), desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                result = plan_synthesis(
                    smiles=target.smiles,
                    model_path=str(model_path),
                    sample_mode="greedy",
                    max_length=max_length,
                    device=device,
                )

                if result["success"]:
                    # Wrap single result in a list to match multi-pathway format
                    pathway = {
                        "synthesis": result["synthesis"],
                        "num_steps": result["num_steps"],
                    }
                    results[target.id] = [pathway]
                    solved_count += 1
                else:
                    results[target.id] = []
                    if result.get("error"):
                        logger.warning(f"Target {target.id} ({target.smiles}): {result['error']}")

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                results[target.id] = []

    return results, solved_count, timer.to_model()


def save_synllama_results(
    results: dict[str, list[dict[str, Any]]],
    runtime: ExecutionStats,
    save_dir: Path,
    bench_path: Path,
    script_name: str,
    benchmark: BenchmarkSet,
    planner_version: str,
    rxn_set: str,
) -> None:
    """Save SynLlama results, execution stats, and manifest.

    Args:
        results: Dictionary mapping target IDs to pathway lists.
        runtime: Execution timing information.
        save_dir: Directory to save outputs.
        bench_path: Path to benchmark definition file.
        script_name: Name of the calling script (for manifest).
        benchmark: Benchmark object (for statistics).
        planner_version: Version of the SynLlama library used.
        rxn_set: Reaction set used ("91rxns" or "115rxns").
    """
    solved_count = sum(1 for routes in results.values() if routes)

    summary = {
        "solved_count": solved_count,
        "total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")

    # Create manifest
    manifest = create_manifest(
        action=script_name,
        sources=[bench_path],
        root_dir=save_dir.parents[2],  # data/retrocast directory
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
        directives={
            "adapter": "synllama",
            "planner_version": planner_version,
            "rxn_set": rxn_set,
            "mode": "raw",
            "raw_results_filename": "results.json.gz",
        },
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")


def main() -> None:
    """Main entry point for running SynLlama raw predictions."""
    parser = argparse.ArgumentParser(
        description="Run SynLlama retrosynthesis with raw generation (no BB reconstruction)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g., mkt-lin-500)",
    )
    parser.add_argument(
        "--rxn-set",
        type=str,
        default="91rxns",
        choices=["91rxns", "115rxns"],
        help="Reaction set to use (default: 91rxns)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Get paths
    paths = get_synllama_paths(args.rxn_set)

    # Validate synllama-data exists
    if not paths.synllama_data.exists():
        raise FileNotFoundError(
            f"synllama-data directory not found at {paths.synllama_data}\n"
            "Please download and extract SynLlama model files to this location."
        )

    if not paths.model_path.exists():
        raise FileNotFoundError(
            f"SynLlama model not found at {paths.model_path}\nPlease ensure the {args.rxn_set} model is available."
        )

    # Load benchmark
    bench_path = paths.benchmarks_dir / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_path)

    # Setup output directory
    folder_name = f"synllama-{PLANNER_VERSION}-raw-{args.rxn_set}"
    save_dir = paths.raw_dir / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Benchmark: {benchmark.name}")
    logger.info(f"Reaction set: {args.rxn_set}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info("Mode: raw generation (no BB reconstruction)")

    # Run predictions
    logger.info("Retrosynthesis starting")
    results, solved_count, runtime = run_synllama_raw_predictions(
        benchmark=benchmark,
        model_path=paths.model_path,
        max_length=1600,
        device=args.device,
    )

    # Save results
    save_synllama_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        script_name="scripts/planning/run-synllama/1-run-synllama-raw.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        rxn_set=args.rxn_set,
    )


if __name__ == "__main__":
    main()
