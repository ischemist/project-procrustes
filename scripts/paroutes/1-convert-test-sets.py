"""
Converts raw PaRoutes data into the standard retrocast format.

This script takes a PaRoutes JSON file (all-routes, n1-routes, or n5-routes),
adapts each route to the standard Route schema, and saves the results along
with a manifest containing file hashes for verification.

Usage:
    uv run scripts/paroutes/1-convert-test-sets.py all-routes
    uv run scripts/paroutes/1-convert-test-sets.py n1-routes
    uv run scripts/paroutes/1-convert-test-sets.py n5-routes
"""

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.io import save_json, save_routes
from retrocast.schemas import Route, TargetInput, _get_retrocast_version
from retrocast.utils.hashing import compute_routes_content_hash, generate_file_hash
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PAROUTES_DIR = BASE_DIR / "data" / "paroutes"
OUTPUT_DIR = PAROUTES_DIR / "processed"

VALID_DATASETS = ["all-routes", "n1-routes", "n5-routes"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PaRoutes data to standard retrocast format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        choices=VALID_DATASETS,
        help="Name of the PaRoutes dataset to convert (without .json.gz extension)",
    )
    return parser.parse_args()


def load_raw_routes(input_file: Path) -> list[dict]:
    """Load raw routes from a gzipped JSON file."""
    logger.info(f"Loading raw routes from {input_file}...")
    try:
        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            all_routes = json.load(f)
        if not isinstance(all_routes, list):
            raise RetroCastException(f"Expected a list of routes in {input_file}, but got {type(all_routes)}")
        logger.info(f"Loaded {len(all_routes):,} total routes.")
        return all_routes
    except (OSError, json.JSONDecodeError) as e:
        raise RetroCastException(f"Failed to load or parse {input_file}: {e}") from e


def adapt_routes(raw_routes: list[dict], dataset_prefix: str) -> dict[str, list[Route]]:
    """Adapt raw routes to the standard Route schema."""
    adapter = PaRoutesAdapter()
    adapted_routes: dict[str, list[Route]] = {}
    failed_count = 0

    pbar = tqdm(enumerate(raw_routes, 1), total=len(raw_routes), desc="Adapting routes")
    for i, raw_route in pbar:
        target_id = f"{dataset_prefix}-{i}"
        try:
            target_smiles = canonicalize_smiles(raw_route["smiles"])
            target_info = TargetInput(id=target_id, smiles=target_smiles)

            # PaRoutes: each input is a single route, so we expect 0 or 1 trees
            routes = list(adapter.adapt(raw_route, target_info))

            if routes:
                adapted_routes[target_id] = routes
        except RetroCastException as e:
            logger.debug(f"Could not process route {i}: {e}")
            failed_count += 1
        except (KeyError, TypeError) as e:
            logger.debug(f"Route {i} has invalid structure: {e}")
            failed_count += 1

    logger.info(f"Successfully adapted {len(adapted_routes)}/{len(raw_routes)} routes ({failed_count} failed)")
    adapter.report_statistics()
    return adapted_routes


def create_manifest(
    dataset_name: str,
    input_file: Path,
    output_file: Path,
    routes: dict[str, list[Route]],
    raw_route_count: int,
) -> dict:
    """Create a manifest with file hashes and statistics."""
    manifest = {
        "dataset": dataset_name,
        "source_file": input_file.name,
        "source_file_hash": generate_file_hash(input_file),
        "output_file": output_file.name,
        "output_file_hash": generate_file_hash(output_file),
        "output_content_hash": compute_routes_content_hash(routes),
        "statistics": {
            "total_raw_routes": raw_route_count,
            "successful_adaptations": len(routes),
            "failed_adaptations": raw_route_count - len(routes),
            "total_routes_saved": sum(len(r) for r in routes.values()),
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "retrocast_version": _get_retrocast_version(),
    }
    return manifest


def main() -> None:
    """Main script execution."""
    args = parse_args()
    dataset_name = args.dataset

    input_file = PAROUTES_DIR / f"{dataset_name}.json.gz"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_name}.json.gz"
    manifest_file = OUTPUT_DIR / f"{dataset_name}-manifest.json"

    # Load and adapt routes
    raw_routes = load_raw_routes(input_file)

    # Extract dataset prefix for target IDs (e.g., "n1" from "n1-routes")
    dataset_prefix = dataset_name.replace("-routes", "")
    adapted_routes = adapt_routes(raw_routes, dataset_prefix)

    if not adapted_routes:
        logger.error("No routes were successfully adapted. No output files will be written.")
        raise SystemExit(1)

    # Save routes
    logger.info(f"Saving adapted routes to {output_file.relative_to(BASE_DIR)}...")
    save_routes(adapted_routes, output_file)

    # Create and save manifest
    manifest = create_manifest(dataset_name, input_file, output_file, adapted_routes, len(raw_routes))
    logger.info(f"Saving manifest to {manifest_file.relative_to(BASE_DIR)}...")
    save_json(manifest, manifest_file)

    # Report summary
    logger.info("\n--- Conversion Summary ---")
    logger.info(f"Input file hash:    {manifest['source_file_hash'][:16]}...")
    logger.info(f"Output file hash:   {manifest['output_file_hash'][:16]}...")
    logger.info(f"Content hash:       {manifest['output_content_hash'][:16]}...")
    logger.info(f"Total routes saved: {manifest['statistics']['total_routes_saved']}")
    logger.info("--- Done ---")


if __name__ == "__main__":
    main()
