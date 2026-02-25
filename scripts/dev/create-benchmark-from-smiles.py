"""
Create a benchmark from a hardcoded list of SMILES (without reference routes).

Usage:
    uv run scripts/dev/create-benchmark-from-smiles.py
    uv run scripts/dev/create-benchmark-from-smiles.py --name nitrazepam --stock-name buyables-stock
    uv run scripts/dev/create-benchmark-from-smiles.py --name test-set --description "Custom test molecules"

Creates a benchmark for pure prediction tasks with no reference routes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import InvalidSmilesError
from retrocast.io import create_manifest, save_json_gz
from retrocast.models.benchmark import create_benchmark, create_benchmark_target
from retrocast.typing import InchiKeyStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[2]
DEF_DIR = BASE_DIR / "data" / "retrocast" / "1-benchmarks" / "definitions"

# Hardcoded SMILES list
SMILES_LIST = [
    "[O-][N+](C1=CC=C2C(C(C3=CC=CC=C3)=NCC(N2)=O)=C1)=O",
]


def main() -> None:
    """Create benchmark from hardcoded SMILES list."""
    parser = argparse.ArgumentParser(description="Create benchmark from SMILES list")
    parser.add_argument(
        "--name",
        default="custom-smiles",
        help="Benchmark name (default: custom-smiles)",
    )
    parser.add_argument(
        "--stock-name",
        default=None,
        help="Stock name to associate with benchmark (e.g., buyables-stock)",
    )
    parser.add_argument(
        "--description",
        default="Custom benchmark from SMILES list",
        help="Benchmark description",
    )
    args = parser.parse_args()

    configure_script_logging()

    logger.info(f"Creating benchmark '{args.name}' with {len(SMILES_LIST)} target(s)")

    # Process SMILES into BenchmarkTargets
    targets = {}
    failed = 0

    for idx, smiles in enumerate(SMILES_LIST, start=1):
        target_id = f"rc-custom-{idx:03d}"

        try:
            # Canonicalize and create target
            canonical_smiles = canonicalize_smiles(smiles)
            inchi_key = get_inchi_key(canonical_smiles)

            target = create_benchmark_target(
                id=target_id,
                smiles=canonical_smiles,
                acceptable_routes=[],  # No reference routes for prediction-only benchmark
                metadata={},
            )

            targets[target_id] = target
            logger.info(f"  {target_id}: {canonical_smiles} → {inchi_key}")

        except InvalidSmilesError as e:
            logger.error(f"  {target_id}: Failed to process '{smiles}': {e}")
            failed += 1
        except Exception as e:
            logger.error(f"  {target_id}: Unexpected error: {e}")
            failed += 1

    if not targets:
        logger.error("No valid targets created. Exiting.")
        return

    # Create benchmark (validation happens inside create_benchmark)
    # Pass empty stock set since we have no routes to validate
    stock: set[InchiKeyStr] = set()
    benchmark = create_benchmark(
        name=args.name,
        targets=targets,
        stock=stock,
        description=args.description,
        stock_name=args.stock_name,
    )

    # Save benchmark
    DEF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEF_DIR / f"{args.name}.json.gz"
    save_json_gz(benchmark, output_path)
    logger.info(f"Saved benchmark to {output_path}")

    # Create manifest
    manifest_path = DEF_DIR / f"{args.name}.manifest.json"
    manifest = create_manifest(
        action="scripts/dev/create-benchmark-from-smiles",
        sources=[],
        outputs=[(output_path, benchmark, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={
            "name": args.name,
            "stock_name": args.stock_name,
            "description": args.description,
        },
        statistics={
            "n_targets": len(targets),
            "n_failed": failed,
        },
    )
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))
    logger.info(f"Saved manifest to {manifest_path}")

    logger.info(f"✓ Created benchmark with {len(targets)} target(s) ({failed} failed)")


if __name__ == "__main__":
    main()
