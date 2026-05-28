from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from retrocast.adapters.errors import adapter_cycle_error
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.base import AdaptMode
from retrocast.v2.models.route import Molecule, Reaction

ReactionAnnotations = Mapping[SmilesStr, Mapping[str, Any]]


# SECTION: Precursor Map Traversal


def build_molecule_from_precursor_map(
    smiles: str,
    precursor_map: Mapping[SmilesStr, Sequence[str]],
    *,
    adapter: str,
    mode: AdaptMode,
    visited: set[SmilesStr] | None = None,
    reaction_annotations: ReactionAnnotations | None = None,
) -> Molecule | None:
    if visited is None:
        visited = set()

    try:
        canon_smiles = canonicalize_smiles(smiles)
    except InvalidSmilesError:
        if mode == "prune":
            return None
        raise

    if canon_smiles in visited:
        raise adapter_cycle_error(adapter, canon_smiles)

    reactant_smiles = precursor_map.get(canon_smiles)
    if reactant_smiles is None:
        return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))

    reactants: list[Molecule] = []
    for reactant in reactant_smiles:
        reactant_molecule = build_molecule_from_precursor_map(
            reactant,
            precursor_map,
            adapter=adapter,
            mode=mode,
            visited=visited | {canon_smiles},
            reaction_annotations=reaction_annotations,
        )
        if reactant_molecule is not None:
            reactants.append(reactant_molecule)

    if not reactants:
        if mode == "prune":
            return None
        raise AdapterLogicError(
            f"{adapter} reaction has no reactants",
            code="adapter.reaction_empty",
            context={"adapter": adapter, "smiles": canon_smiles},
        )

    annotations = dict(reaction_annotations.get(canon_smiles, {})) if reaction_annotations is not None else {}
    return Molecule(
        smiles=canon_smiles,
        inchikey=get_inchi_key(canon_smiles),
        product_of=Reaction(reactants=reactants, annotations=annotations),
    )
