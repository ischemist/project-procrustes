"""
Shared test helper functions for creating synthetic routes and molecules.

These utilities enable testing route topology and chemistry without complex real-world data.
"""

import hashlib

from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr


def _synthetic_inchikey(smiles: str) -> str:
    """
    Generate a deterministic fake InchiKey from SMILES for testing.
    Format mimics real InchiKey structure: XXXXXXXXXXXXXX-XXXXXXXXXX-X
    """
    h = hashlib.sha256(smiles.encode()).hexdigest().upper()
    return f"{h[:14]}-{h[14:24]}-N"


def _make_leaf_molecule(smiles: str) -> Molecule:
    """Create a leaf molecule (no synthesis step) from SMILES."""
    return Molecule(
        smiles=SmilesStr(smiles),
        inchikey=InchiKeyStr(_synthetic_inchikey(smiles)),
        synthesis_step=None,
    )


def _make_simple_route(target_smiles: str, leaf_smiles: str, rank: int = 1) -> Route:
    """Create a simple one-step route: target <- leaf."""
    leaf = _make_leaf_molecule(leaf_smiles)
    target = Molecule(
        smiles=SmilesStr(target_smiles),
        inchikey=InchiKeyStr(_synthetic_inchikey(target_smiles)),
        synthesis_step=ReactionStep(reactants=[leaf]),
    )
    return Route(target=target, rank=rank)


def _make_two_step_route(target_smiles: str, intermediate_smiles: str, leaf_smiles: str, rank: int = 1) -> Route:
    """Create a two-step route: target <- intermediate <- leaf."""
    leaf = _make_leaf_molecule(leaf_smiles)
    intermediate = Molecule(
        smiles=SmilesStr(intermediate_smiles),
        inchikey=InchiKeyStr(_synthetic_inchikey(intermediate_smiles)),
        synthesis_step=ReactionStep(reactants=[leaf]),
    )
    target = Molecule(
        smiles=SmilesStr(target_smiles),
        inchikey=InchiKeyStr(_synthetic_inchikey(target_smiles)),
        synthesis_step=ReactionStep(reactants=[intermediate]),
    )
    return Route(target=target, rank=rank)
