"""
Casts raw PaRoutes n1/n5 JSON files into the canonical BenchmarkSet format.

Usage:
    uv run scripts/paroutes/01-cast-paroutes.py
"""

from pathlib import Path

from tqdm import tqdm

from retrocast import adapt_single_route
from retrocast.chem import canonicalize_smiles
from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_raw_paroutes_list, load_stock_file
from retrocast.io.manifests import create_manifest
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import TargetInput
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "0-paroutes"  # Where raw n1/n5 json.gz live
DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"
stock_dir = BASE_DIR / "data" / "1-benchmarks" / "stocks"

DATASETS = ["n1", "n5"]
buyables = load_stock_file(stock_dir / "buyables-stock.txt")


def process_dataset(name: str):
    raw_path = RAW_DIR / f"{name}-routes.json.gz"
    out_path = DEF_DIR / f"paroutes-{name}-full-buyables.json.gz"

    logger.info(f"Processing {name}...")
    raw_list = load_raw_paroutes_list(raw_path)

    benchmark = BenchmarkSet(
        name=f"paroutes-{name}-full", description=f"Full raw import of PaRoutes {name} set.", stock_name=f"{name}-stock"
    )

    failures = 0

    for i, raw_item in tqdm(enumerate(raw_list), total=len(raw_list), desc=f"Casting {name}"):
        # Generate stable ID based on index
        # n5-00001, n5-00002...
        target_id = f"{name}-{i + 1:05d}"

        # 1. Canonicalize SMILES
        smiles = canonicalize_smiles(raw_item["smiles"])

        # 2. Adapt the Route
        # (We construct a temporary TargetIdentity for the adapter)
        target_input = TargetInput(id=target_id, smiles=smiles)
        route = adapt_single_route(raw_item, target_input, "paroutes")

        if not route:
            failures += 1
            continue

        is_solved = is_route_solved(route, buyables)
        if not is_solved:
            continue

        # 3. Create BenchmarkTarget
        # Calculate metadata immediately so we can query it later
        target = BenchmarkTarget(
            id=target_id,
            smiles=smiles,
            ground_truth=route,
            route_length=route.length,
            is_convergent=route.has_convergent_reaction,
        )

        benchmark.targets[target_id] = target

    logger.info(f"Created {len(benchmark.targets)} targets. {failures} failed.")
    save_json_gz(benchmark, out_path)

    manifest_path = DEF_DIR / f"paroutes-{name}.manifest.json"
    manifest = create_manifest(
        action="scripts/paroutes/01-cast-paroutes",
        sources=[raw_path],
        outputs=[(out_path, benchmark)],
        parameters={"dataset": name},
        statistics={"n_targets": len(benchmark.targets), "n_failures": failures},
    )
    # Save Manifest (plain JSON, not gzipped, so it's readable)
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Manifest saved to {manifest_path}")


def main():
    DEF_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        process_dataset(ds)


if __name__ == "__main__":
    main()
