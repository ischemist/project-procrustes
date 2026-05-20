"""
Check reaction direction in pickle file by comparing first/last molecules to targets.

Usage:
    uv run scripts/curation/uspto-190/check-reaction-direction.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from retrocast.chem import get_inchi_key
from retrocast.exceptions import InvalidSmilesError
from retrocast.utils.logging import configure_script_logging

logger = logging.getLogger("retrocast")


def load_target_inchi_keys(csv_path: Path) -> set[str]:
    """Load target InChI keys from CSV file (SMILES in second column)."""
    inchi_keys = set()
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            smiles = line.strip().split(",")[1]
            try:
                inchi_keys.add(get_inchi_key(smiles))
            except InvalidSmilesError:
                logger.warning(f"Could not parse target SMILES: {smiles}")
    return inchi_keys


def get_first_molecule(reaction_smiles: str) -> str:
    """Extract first molecule from reaction SMILES (format: "reactants>>products")."""
    return reaction_smiles.split(">>")[0].split(".")[0]


def get_last_molecule(reaction_smiles: str) -> str:
    """Extract first product molecule from reaction SMILES."""
    return reaction_smiles.split(">>")[1].split(".")[0]


def check_route_direction(pickle_path: Path, targets_csv_path: Path) -> None:
    """Determine if routes are retrosynthetic or forward by InChI key matching."""
    with open(pickle_path, "rb") as f:
        routes = pickle.load(f)

    target_keys = load_target_inchi_keys(targets_csv_path)
    logger.info(f"Loaded {len(routes)} routes and {len(target_keys)} targets")

    first_reactant_matches = 0
    last_product_matches = 0

    for route_steps in routes:
        try:
            first_key = get_inchi_key(get_first_molecule(route_steps[0]))
            if first_key in target_keys:
                first_reactant_matches += 1

            last_key = get_inchi_key(get_last_molecule(route_steps[-1]))
            if last_key in target_keys:
                last_product_matches += 1
        except InvalidSmilesError:
            pass

    logger.info(f"First reactants matching targets: {first_reactant_matches}/{len(routes)}")
    logger.info(f"Last products matching targets: {last_product_matches}/{len(routes)}")

    if first_reactant_matches > last_product_matches:
        logger.info("✓ Routes are RETROSYNTHETIC (target → precursors)")
    elif last_product_matches > first_reactant_matches:
        logger.info("✓ Routes are FORWARD (precursors → target)")
    else:
        logger.warning("✗ Cannot determine direction")


if __name__ == "__main__":
    configure_script_logging()
    check_route_direction(Path("data/0-assets/routes_possible_test_hard.pkl"), Path("data/uspto-190.csv"))
