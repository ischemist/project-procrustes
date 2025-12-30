import logging
from typing import Any, Protocol

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError
from retrocast.models.chem import Molecule, ReactionStep
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

PrecursorMap = dict[SmilesStr, list[SmilesStr]]


# --- pattern a: bipartite graph recursor ---


class BipartiteMolNode(Protocol):
    """
    a protocol defining the shape of a raw molecule node from bipartite-graph-style outputs.

    note: we use `@property` to define the members, making them read-only (covariant).
    this allows concrete pydantic models with more specific types (e.g., `Literal['mol']`)
    to correctly match the protocol's `type: str` without mypy raising an invariance error.
    """

    @property
    def type(self) -> str: ...

    @property
    def smiles(self) -> str: ...

    @property
    def children(self) -> list[Any]: ...

    @property
    def in_stock(self) -> bool: ...


class BipartiteRxnNode(Protocol):
    """
    a protocol defining the shape of a raw reaction node from bipartite-graph-style outputs.
    see `BipartiteMolNode` docstring for explanation of `@property` usage.
    """

    @property
    def type(self) -> str: ...

    @property
    def children(self) -> list[BipartiteMolNode]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


# --- pattern b: precursor map recursor ---

# --- Schema Helpers (Route/Molecule/ReactionStep) ---


def build_molecule_from_precursor_map(
    smiles: SmilesStr,
    precursor_map: PrecursorMap,
    visited: set[SmilesStr] | None = None,
    ignore_stereo: bool = False,
) -> Molecule:
    """
    Recursively builds a `Molecule` from a precursor map, with cycle detection.
    This is the new schema version for models like retro*, dreamretro, etc.

    Args:
        smiles: The SMILES string of the current molecule.
        precursor_map: A dict mapping product SMILES to list of reactant SMILES.
        visited: Set of SMILES already visited (for cycle detection).
        ignore_stereo: If True, stereochemistry is stripped during SMILES canonicalization.
            Defaults to False.

    Returns:
        A Molecule object representing this node and its synthesis tree.
    """
    if visited is None:
        visited = set()

    # Cycle detection
    if smiles in visited:
        logger.warning(f"Cycle detected in route graph involving smiles: {smiles}. Treating as a leaf node.")
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=None,
            metadata={},
        )

    new_visited = visited | {smiles}
    is_leaf = smiles not in precursor_map

    if is_leaf:
        # This is a starting material (leaf node)
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=None,
            metadata={},
        )

    # Build reactants recursively
    reactant_smiles_list = precursor_map[smiles]
    reactant_molecules: list[Molecule] = []

    for reactant_smi in reactant_smiles_list:
        reactant_mol = build_molecule_from_precursor_map(
            smiles=reactant_smi,
            precursor_map=precursor_map,
            visited=new_visited,
            ignore_stereo=ignore_stereo,
        )
        reactant_molecules.append(reactant_mol)

    # Create the reaction step
    synthesis_step = ReactionStep(
        reactants=reactant_molecules,
        mapped_smiles=None,
        reagents=None,
        solvents=None,
        metadata={},
    )

    # Create the molecule with its synthesis step
    return Molecule(
        smiles=smiles,
        inchikey=get_inchi_key(smiles),
        synthesis_step=synthesis_step,
        metadata={},
    )


def build_molecule_from_bipartite_node(raw_mol_node: BipartiteMolNode, ignore_stereo: bool = False) -> Molecule:
    """
    Recursively builds a `Molecule` from a raw, validated bipartite graph node.
    This is the new schema version for models like aizynthfinder, synplanner, etc.

    Args:
        raw_mol_node: A raw molecule node following the BipartiteMolNode protocol.
        ignore_stereo: If True, stereochemistry is stripped during SMILES canonicalization.
            Defaults to False.

    Returns:
        A Molecule object representing this node and its synthesis tree.
    """
    if raw_mol_node.type != "mol":
        raise AdapterLogicError(f"Expected node type 'mol' but got '{raw_mol_node.type}'")

    canon_smiles = canonicalize_smiles(raw_mol_node.smiles, ignore_stereo=ignore_stereo)
    is_leaf = raw_mol_node.in_stock or not bool(raw_mol_node.children)

    if is_leaf:
        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=None,
            metadata={},
        )

    # In a valid tree, a molecule has at most one reaction leading to it
    if len(raw_mol_node.children) > 1:
        logger.warning(
            f"Molecule {canon_smiles} has multiple child reactions in raw output; only the first is used in a tree."
        )

    raw_reaction_node: BipartiteRxnNode = raw_mol_node.children[0]
    if raw_reaction_node.type != "reaction":
        raise AdapterLogicError("Child of molecule node was not a reaction node")

    # Build reactants recursively
    reactant_molecules: list[Molecule] = []
    for reactant_mol_input in raw_reaction_node.children:
        reactant_mol = build_molecule_from_bipartite_node(raw_mol_node=reactant_mol_input, ignore_stereo=ignore_stereo)
        reactant_molecules.append(reactant_mol)

    # Extract template and mapped_smiles from metadata if available
    rxn_metadata = raw_reaction_node.metadata
    template = rxn_metadata.get("template") if rxn_metadata else None
    mapped_smiles = rxn_metadata.get("mapped_reaction_smiles") if rxn_metadata else None

    # Create the reaction step
    synthesis_step = ReactionStep(
        reactants=reactant_molecules,
        mapped_smiles=mapped_smiles,
        template=template,
        reagents=None,
        solvents=None,
        metadata=rxn_metadata if rxn_metadata else {},
    )

    return Molecule(
        smiles=canon_smiles,
        inchikey=get_inchi_key(canon_smiles),
        synthesis_step=synthesis_step,
        metadata={},
    )
