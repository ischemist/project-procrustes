"""
Converts raw PaRoutes data into the standard retrocast format.

This script takes a PaRoutes JSON file (all-routes, n1-routes, or n5-routes),
adapts each route to the standard Route schema, and saves the results along
with a manifest containing file hashes for verification.

Usage:
    uv run scripts/paroutes/1-cast-raw-routes.py all-routes
    uv run scripts/paroutes/1-cast-raw-routes.py n5-routes
    uv run scripts/paroutes/1-cast-raw-routes.py n1-routes --exclude-signatures-from data/paroutes/processed/n5-routes.json.gz
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast import Route, TargetInput, adapt_single_route, deduplicate_routes
from retrocast.curation import create_manifest, filter_routes_by_signature, get_route_signatures
from retrocast.domain.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.io import load_raw_routes, load_routes, save_json, save_routes, save_smiles_index
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
PAROUTES_DIR = BASE_DIR / "data" / "paroutes"
OUTPUT_DIR = PAROUTES_DIR / "processed"

VALID_DATASETS = ["all-routes", "n1-routes", "n5-routes"]


def adapt_routes(raw_routes: list[dict], dataset_prefix: str) -> dict[str, list[Route]]:
    """Cast raw routes to the standard Route schema."""
    adapted_routes: dict[str, Route] = {}
    routes: list[Route] = []
    failed_count = 0

    pbar = tqdm(enumerate(raw_routes, 1), total=len(raw_routes), desc="Casting routes")
    for i, raw_route in pbar:
        target_id = f"{dataset_prefix}-{i}"
        try:
            target_smiles = canonicalize_smiles(raw_route["smiles"])
            target = TargetInput(id=target_id, smiles=target_smiles)

            # PaRoutes: each input is a single route, so we expect 0 or 1 trees
            route = adapt_single_route(raw_route, target, "paroutes")
            if route:
                routes.append(route)
        except RetroCastException as e:
            logger.debug(f"Could not process route {i}: {e}")
            failed_count += 1
        except (KeyError, TypeError) as e:
            logger.debug(f"Route {i} has invalid structure: {e}")
            failed_count += 1

    unique_routes = deduplicate_routes(routes)
    adapted_routes = {f"{dataset_prefix}-{i:05d}": [route] for i, route in enumerate(unique_routes)}

    logger.info(f"Successfully adapted {len(adapted_routes)}/{len(raw_routes)} routes ({failed_count} failed)")
    return adapted_routes


def build_smiles_index(routes: dict[str, list[Route]]) -> dict[str, str]:
    """Build a SMILES to target ID index from routes."""
    index = {}
    for target_id, route_list in routes.items():
        if route_list:
            # All routes for a target should have the same target SMILES
            smiles = route_list[0].target.smiles
            index[smiles] = target_id
    return index


def main() -> None:
    """Main script execution."""
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
    parser.add_argument(
        "--exclude-signatures-from",
        type=Path,
        help="Path to a processed routes file. Routes with matching signatures will be excluded.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    input_file = PAROUTES_DIR / f"{dataset_name}.json.gz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_name}.json.gz"
    manifest_file = OUTPUT_DIR / f"{dataset_name}-manifest.json"
    index_file = OUTPUT_DIR / f"{dataset_name}-smiles-index.json.gz"

    raw_routes = load_raw_routes(input_file)

    dataset_prefix = dataset_name.replace("-routes", "")
    adapted_routes = adapt_routes(raw_routes, dataset_prefix)

    # Filter out routes that match signatures from another dataset
    exclude_signatures: set[str] = set()
    if args.exclude_signatures_from:
        logger.info(f"Loading routes to exclude from {args.exclude_signatures_from}...")
        exclude_routes = load_routes(args.exclude_signatures_from)
        exclude_signatures = get_route_signatures(exclude_routes)
        logger.info(f"Found {len(exclude_signatures)} signatures to exclude.")

        original_count = sum(len(r) for r in adapted_routes.values())
        adapted_routes = filter_routes_by_signature(adapted_routes, exclude_signatures)
        filtered_count = sum(len(r) for r in adapted_routes.values())
        logger.info(f"Filtered out {original_count - filtered_count} duplicate routes.")

    if not adapted_routes:
        logger.error("No routes were successfully adapted. No output files will be written.")
        raise SystemExit(1)

    logger.info(f"Saving adapted routes to {output_file.relative_to(BASE_DIR)}...")
    save_routes(adapted_routes, output_file)

    # Save SMILES to target ID index
    smiles_index = build_smiles_index(adapted_routes)
    save_smiles_index(smiles_index, index_file)

    statistics = {
        "n_routes_source": len(raw_routes),
        "n_routes_saved": sum(len(r) for r in adapted_routes.values()),
        "n_routes_excluded": len(exclude_signatures) if exclude_signatures else 0,
    }
    manifest = create_manifest(dataset_name, input_file, output_file, adapted_routes, statistics)
    save_json(manifest, manifest_file)


if __name__ == "__main__":
    main()
