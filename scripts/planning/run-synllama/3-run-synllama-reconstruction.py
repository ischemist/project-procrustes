"""
Run SynLlama retrosynthesis predictions on a batch of targets with custom BB reconstruction.

This script processes targets from a benchmark using SynLlama's LLM-based synthesis
planning with custom building block reconstruction and saves results in a structured
format matching other prediction scripts.

Example usage:
    uv run --directory scripts/planning/run-synllama 3-run-synllama-reconstruction.py --benchmark mkt-lin-500
    uv run --directory scripts/planning/run-synllama 3-run-synllama-reconstruction.py --benchmark random-n5-2-seed=20251030 --effort high
    uv run --directory scripts/planning/run-synllama 3-run-synllama-reconstruction.py --benchmark mkt-lin-500 --rxn-set 115rxns

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Custom BB indices must exist at: synllama-data/inference/reconstruction/{rxn_set}/custom_bb_indices/{stock_name}/
Results are saved to: data/retrocast/2-raw/synllama-{version}-reconstruction-{rxn_set}[-{effort}]/{benchmark_name}/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synllama.api import plan_synthesis_with_reconstruction
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
    rxn_embedding_path: Path
    custom_bb_base: Path
    stocks_dir: Path
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
    rxn_embedding_path = synllama_data / "inference" / "reconstruction" / rxn_set / "rxn_embeddings"
    custom_bb_base = synllama_data / "inference" / "reconstruction" / rxn_set / "custom_bb_indices"

    return SynLlamaPaths(
        synllama_data=synllama_data,
        model_path=model_path,
        rxn_embedding_path=rxn_embedding_path,
        custom_bb_base=custom_bb_base,
        stocks_dir=paths["stocks"],
        benchmarks_dir=paths["benchmarks"],
        raw_dir=paths["raw"],
    )


def load_benchmark_and_validate(benchmark_name: str, paths: SynLlamaPaths) -> tuple[BenchmarkSet, Path, Path]:
    """Load benchmark definition and validate stock has BB indices.

    Args:
        benchmark_name: Name of the benchmark (without extension).
        paths: SynLlamaPaths instance with resource locations.

    Returns:
        Tuple of (benchmark, bench_path, custom_bb_index_path).

    Raises:
        AssertionError: If benchmark has no stock_name defined.
        FileNotFoundError: If custom BB indices don't exist for the stock.
    """
    bench_path = paths.benchmarks_dir / f"{benchmark_name}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {benchmark_name}"

    # Validate custom BB indices exist
    custom_bb_index_path = paths.custom_bb_base / benchmark.stock_name
    if not custom_bb_index_path.exists():
        raise FileNotFoundError(
            f"Custom BB indices not found for stock '{benchmark.stock_name}' at {custom_bb_index_path}\n"
            f"Please run 2-build-bb-indices.py first:\n"
            f"  uv run --directory scripts/planning/run-synllama 2-build-bb-indices.py --stock {benchmark.stock_name}"
        )

    return benchmark, bench_path, custom_bb_index_path


def run_synllama_predictions(
    benchmark: BenchmarkSet,
    model_path: Path,
    rxn_embedding_path: Path,
    custom_bb_index_path: Path,
    top_n: int,
    k: int,
    n_stacks: int,
    device: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int, ExecutionStats]:
    """Run SynLlama reconstruction over all benchmark targets.

    Args:
        benchmark: Benchmark containing targets to process.
        model_path: Path to SynLlama model directory.
        rxn_embedding_path: Path to reaction embeddings directory.
        custom_bb_index_path: Path to custom BB indices directory.
        top_n: Number of top pathways to return per target.
        k: Number of similar BBs to search per reactant.
        n_stacks: Maximum stacks during reconstruction.
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
                result = plan_synthesis_with_reconstruction(
                    smiles=target.smiles,
                    model_path=str(model_path),
                    rxn_embedding_path=str(rxn_embedding_path),
                    custom_bb_index_path=str(custom_bb_index_path),
                    sample_mode="greedy",
                    top_n=top_n,
                    k=k,
                    n_stacks=n_stacks,
                    max_length=1600,
                    device=device,
                )

                if result["success"] and result["pathways"]:
                    results[target.id] = result["pathways"]
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
    custom_bb_index_path: Path,
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
        custom_bb_index_path: Path to custom BB indices directory.
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

    # Create manifest with custom BB indices path as source
    manifest = create_manifest(
        action=script_name,
        sources=[bench_path, custom_bb_index_path],
        root_dir=save_dir.parents[2],  # data/retrocast directory
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
        directives={
            "adapter": "synllama",
            "planner_version": planner_version,
            "rxn_set": rxn_set,
            "raw_results_filename": "results.json.gz",
        },
    )

    with open(save_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")


def main() -> None:
    """Main entry point for running SynLlama predictions."""
    parser = argparse.ArgumentParser(description="Run SynLlama retrosynthesis with custom BB reconstruction")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g., mkt-lin-500)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="normal",
        choices=["normal", "high"],
        help="Search effort level: normal (k=5, n_stacks=25) or high (k=10, n_stacks=50)",
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

    if not paths.rxn_embedding_path.exists():
        raise FileNotFoundError(
            f"Reaction embeddings not found at {paths.rxn_embedding_path}\n"
            f"Please ensure the {args.rxn_set} reaction embeddings are available."
        )

    # Load benchmark and validate BB indices exist
    benchmark, bench_path, custom_bb_index_path = load_benchmark_and_validate(args.benchmark, paths)

    # Setup output directory
    folder_name = (
        f"synllama-{PLANNER_VERSION}-reconstruction-{args.rxn_set}"
        if args.effort == "normal"
        else f"synllama-{PLANNER_VERSION}-reconstruction-{args.rxn_set}-{args.effort}"
    )
    save_dir = paths.raw_dir / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Benchmark: {benchmark.name}")
    logger.info(f"Stock: {benchmark.stock_name}")
    logger.info(f"Reaction set: {args.rxn_set}")
    logger.info(f"Effort: {args.effort}")
    logger.info(f"Device: {args.device or 'auto-detect'}")

    # Set effort parameters
    if args.effort == "high":
        top_n, k, n_stacks = 10, 10, 50
    else:
        top_n, k, n_stacks = 5, 5, 25

    logger.info(f"Parameters: top_n={top_n}, k={k}, n_stacks={n_stacks}")

    # Run predictions
    logger.info("Retrosynthesis starting")
    results, solved_count, runtime = run_synllama_predictions(
        benchmark=benchmark,
        model_path=paths.model_path,
        rxn_embedding_path=paths.rxn_embedding_path,
        custom_bb_index_path=custom_bb_index_path,
        top_n=top_n,
        k=k,
        n_stacks=n_stacks,
        device=args.device,
    )

    # Save results
    save_synllama_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        custom_bb_index_path=custom_bb_index_path,
        script_name="scripts/planning/run-synllama/3-run-synllama-reconstruction.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        rxn_set=args.rxn_set,
    )


if __name__ == "__main__":
    main()
