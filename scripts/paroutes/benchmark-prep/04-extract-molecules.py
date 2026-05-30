"""
Extracts all unique molecules from the full PaRoutes dataset into a CSV file.

Each molecule is identified by its InChIKey and canonical SMILES, and tagged
with membership in the n1, n5, and evaluation subsets.

Usage:
    uv run scripts/paroutes/benchmark-prep/04-extract-molecules.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from retrocast.adapters.paroutes import PaRoutesAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import RetroCastException
from retrocast.io import load_benchmark, load_json_artifact
from retrocast.models import Molecule, Target
from retrocast.typing import InChIKeyStr, InchiKeyStr, SmilesStr
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


def get_all_molecules(mol: Molecule) -> list[Molecule]:
    """Recursively collect all molecules from a synthesis tree."""
    result = [mol]
    if mol.product_of:
        for reactant in mol.product_of.reactants:
            result.extend(get_all_molecules(reactant))
    return result


def collect_inchikeys_from_benchmark(path: Path) -> set[InchiKeyStr]:
    """Load a benchmark file and return the set of all molecule InChIKeys from primary routes."""
    benchmark = load_benchmark(path)
    inchikeys: set[InchiKeyStr] = set()
    for target in benchmark.targets.values():
        if not target.acceptable_routes:
            continue
        route = target.acceptable_routes[0]
        for mol in get_all_molecules(route.target):
            inchikeys.add(mol.inchikey)
    return inchikeys


def main():
    configure_script_logging()
    parser = argparse.ArgumentParser(description="extract unique molecules from the full PaRoutes dataset.")
    parser.parse_args()

    raw_path = RAW_DIR / "all-routes.json.gz"
    out_path = RAW_DIR / "all-molecules.csv"

    # --- Step 1: Cast all routes and extract molecules ---
    logger.info(f"Loading raw PaRoutes data from {raw_path}...")
    adapter = PaRoutesAdapter()
    raw_entries = list(adapter.iter_raw_routes(load_json_artifact(raw_path)))
    logger.info(f"Loaded {len(raw_entries)} raw routes.")

    # Master dict: inchikey -> smiles
    molecules: dict[InchiKeyStr, str] = {}
    failures = 0

    for source_order, entry in tqdm(enumerate(raw_entries, start=1), total=len(raw_entries), desc="Casting routes"):
        raw_item = entry.payload
        row_index = entry.source_row_index or source_order
        target_id = f"all-{row_index:06d}"
        smiles = canonicalize_smiles(raw_item["smiles"])

        target_input = Target(id=target_id, smiles=SmilesStr(smiles), inchikey=InChIKeyStr(get_inchi_key(smiles)))
        try:
            route = adapter.cast(raw_item, target=target_input)
        except RetroCastException as exc:
            logger.warning("failed to cast route %s: %s [%s]", target_id, exc, exc.code)
            failures += 1
            continue

        for mol in get_all_molecules(route.target):
            if mol.inchikey not in molecules:
                molecules[mol.inchikey] = mol.smiles

    logger.info(
        f"Casting complete. {len(molecules)} unique molecules from {len(raw_entries) - failures} routes ({failures} rejected)."
    )

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
    with open(out_path, "w", newline="", encoding="utf-8") as f:
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
