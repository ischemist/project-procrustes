"""
Extracts all unique molecules from the full PaRoutes dataset into a CSV file.

Each molecule is identified by its InChIKey and canonical SMILES, and tagged
with membership in the n1, n5, and evaluation subsets.

Usage:
    uv run scripts/paroutes/benchmark-prep/04-extract-molecules.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.io import load_benchmark, load_raw_paroutes_list
from retrocast.models.chem import Molecule, Route, TargetInput
from retrocast.typing import InchiKeyStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "retrocast" / "0-assets" / "paroutes"
DEF_DIR = BASE_DIR / "data" / "retrocast" / "1-benchmarks" / "definitions"

# Subset benchmark files -> tag column names
SUBSETS: dict[str, str] = {
    "paroutes-n1-full-pruned.json.gz": "n1",
    "paroutes-n5-full-pruned.json.gz": "n5",
    "mkt-cnv-160.json.gz": "mkt_cnv_160",
    "mkt-lin-500.json.gz": "mkt_lin_500",
    "ref-lng-84.json.gz": "ref_lng_84",
    "ref-lin-600.json.gz": "ref_lin_600",
    "ref-cnv-400.json.gz": "ref_cnv_400",
}


def get_all_molecules(mol: Molecule) -> set[Molecule]:
    """Recursively collect all molecules from a synthesis tree."""
    result = {mol}
    if mol.synthesis_step:
        for reactant in mol.synthesis_step.reactants:
            result.update(get_all_molecules(reactant))
    return result


def collect_inchikeys_from_benchmark(path: Path) -> set[InchiKeyStr]:
    """Load a benchmark file and return the set of all molecule InChIKeys from primary routes."""
    benchmark = load_benchmark(path)
    inchikeys: set[InchiKeyStr] = set()
    for target in benchmark.targets.values():
        route = target.primary_route
        if route is None:
            continue
        for mol in get_all_molecules(route.target):
            inchikeys.add(mol.inchikey)
    return inchikeys


def main():
    configure_script_logging()

    raw_path = RAW_DIR / "all-routes.json.gz"
    out_path = RAW_DIR / "all-molecules.csv"

    # --- Step 1: Cast all routes and extract molecules ---
    logger.info(f"Loading raw PaRoutes data from {raw_path}...")
    raw_list = load_raw_paroutes_list(raw_path)
    logger.info(f"Loaded {len(raw_list)} raw routes.")

    adapter = PaRoutesAdapter()
    # Master dict: inchikey -> smiles
    molecules: dict[InchiKeyStr, str] = {}
    failures = 0

    for i, raw_item in tqdm(enumerate(raw_list), total=len(raw_list), desc="Casting routes"):
        target_id = f"all-{i + 1:06d}"
        smiles = canonicalize_smiles(raw_item["smiles"])

        target_input = TargetInput(id=target_id, smiles=smiles)
        route: Route | None = next(adapter.cast(raw_item, target_input), None)

        if route is None:
            failures += 1
            continue

        for mol in get_all_molecules(route.target):
            if mol.inchikey not in molecules:
                molecules[mol.inchikey] = mol.smiles

    logger.info(
        f"Casting complete. {len(molecules)} unique molecules from {len(raw_list) - failures} routes ({failures} rejected)."
    )
    adapter.report_statistics()

    # --- Step 2: Build tag sets from subset benchmarks ---
    tag_columns = list(SUBSETS.values())
    tag_sets: dict[str, set[InchiKeyStr]] = {}

    for filename, tag in SUBSETS.items():
        path = DEF_DIR / filename
        if not path.exists():
            logger.warning(f"Subset file not found, skipping tag '{tag}': {path}")
            continue
        logger.info(f"Loading subset '{tag}' from {filename}...")
        tag_sets[tag] = collect_inchikeys_from_benchmark(path)
        logger.info(f"  {len(tag_sets[tag])} unique molecules in '{tag}'.")

    # --- Step 3: Write CSV ---
    logger.info(f"Writing {len(molecules)} molecules to {out_path}...")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["inchikey", "smiles"] + tag_columns)

        for inchikey, smiles in sorted(molecules.items()):
            tags = [1 if inchikey in tag_sets.get(tag, set()) else 0 for tag in tag_columns]
            writer.writerow([inchikey, smiles] + tags)

    logger.info(f"Done. CSV written to {out_path}")

    # --- Summary stats ---
    for tag in tag_columns:
        if tag in tag_sets:
            in_master = len(tag_sets[tag] & set(molecules.keys()))
            only_in_subset = len(tag_sets[tag] - set(molecules.keys()))
            logger.info(
                f"  Tag '{tag}': {in_master} molecules tagged ({only_in_subset} in subset but not in master set)"
            )


if __name__ == "__main__":
    main()
