"""
Script to re-export benchmarks keeping only single (primary) acceptable route to calculate original Top-K accuracy.

This script:
1. Loads each benchmark from data/1-benchmarks/definitions/
2. Re-exports them to data/1-benchmarks/definitions/re-export/ with computed hashes
3. Preserves original files untouched

"""

from pathlib import Path

from retrocast.io.blob import save_json_gz
from retrocast.io.data import load_benchmark
from retrocast.io.provenance import create_manifest
from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()


def main():
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    definitions_dir = project_root / "data" / "1-benchmarks" / "definitions"
    reexport_dir = definitions_dir / "single-gt"

    logger.info(f"Source directory: {definitions_dir}")
    logger.info(f"Target directory: {reexport_dir}")

    # Create re-export directory if it doesn't exist
    reexport_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created re-export directory: {reexport_dir}")

    # Find all benchmark files (*.json.gz, excluding manifest files)
    benchmark_files = sorted(definitions_dir.glob("*.json.gz"))
    benchmark_files = [f for f in benchmark_files if not f.name.endswith(".manifest.json")]

    logger.info(f"Found {len(benchmark_files)} benchmark files to process")

    # Process each benchmark
    for benchmark_file in benchmark_files:
        try:
            logger.info(f"\nProcessing: {benchmark_file.name}")

            # Load the benchmark
            benchmark = load_benchmark(benchmark_file)
            logger.info(f"  Loaded benchmark '{benchmark.name}' with {len(benchmark.targets)} targets")

            # Keep only the first acceptable route for each target
            for target in benchmark.targets.values():
                if target.acceptable_routes:
                    target.acceptable_routes = [target.acceptable_routes[0]]
            route_count = sum(len(target.acceptable_routes) for target in benchmark.targets.values())

            # Re-export with computed hashes
            output_file = reexport_dir / benchmark_file.name.replace(".json.gz", "-single-gt.json.gz")
            save_json_gz(benchmark, output_file)
            logger.info(f"  Saved to: {output_file.name}")

            # Create new manifest file
            # The manifest file name is: {basename}.manifest.json (not .json.gz.manifest.json)
            # So for "mkt-cnv-160.json.gz" the manifest is "mkt-cnv-160.manifest.json"
            benchmark_basename = benchmark_file.name.replace(".json.gz", "-single-gt")
            manifest_path = reexport_dir / f"{benchmark_basename}.manifest.json"

            manifest = create_manifest(
                action="scripts/migrations/02-reexport-benchmarks-with-single-gt",
                sources=[benchmark_file],
                outputs=[(output_file, benchmark, "benchmark")],
                root_dir=project_root / "data",
                parameters={"migration": "keep-one-gt-route", "original_file": benchmark_file.name},
                statistics={
                    "n_targets": len(benchmark.targets),
                    "n_ground_truth_routes": route_count,
                },
            )

            with open(manifest_path, "w") as f:
                f.write(manifest.model_dump_json(indent=2))
            logger.info(f"  Created manifest: {manifest_path.name}")

            logger.info(f"  ✓ Successfully processed {benchmark_file.name}")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {benchmark_file.name}: {e}")
            raise

    logger.info(f"\n{'=' * 60}")
    logger.info("Migration complete!")
    logger.info(f"Processed {len(benchmark_files)} benchmark files")
    logger.info(f"Re-exported files are in: {reexport_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
