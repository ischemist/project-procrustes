from typing import Any, Protocol

from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import MoleculeNode, ReactionNode
from retrocast.exceptions import AdapterLogicError
from retrocast.typing import ReactionSmilesStr, SmilesStr
from retrocast.utils.hashing import generate_molecule_hash
from retrocast.utils.logging import logger

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


def build_tree_from_bipartite_node(raw_mol_node: BipartiteMolNode, path_prefix: str) -> MoleculeNode:
    """
    recursively builds a canonical `MoleculeNode` from a raw, validated,
    bipartite graph node (e.g., from aizynthfinder, synplanner).
    """
    if raw_mol_node.type != "mol":
        raise AdapterLogicError(f"expected node type 'mol' but got '{raw_mol_node.type}' at path {path_prefix}")

    canon_smiles = canonicalize_smiles(raw_mol_node.smiles)
    is_starting_mat = raw_mol_node.in_stock or not bool(raw_mol_node.children)
    reactions = []

    if raw_mol_node.children:
        # in a valid tree, a molecule has at most one reaction leading to it.
        if len(raw_mol_node.children) > 1:
            logger.warning(
                f"molecule {canon_smiles} has multiple child reactions in raw output; only the first is used in a tree."
            )

        raw_reaction_node: BipartiteRxnNode = raw_mol_node.children[0]
        if raw_reaction_node.type != "reaction":
            raise AdapterLogicError(f"child of molecule node was not a reaction node at path {path_prefix}")

        reactants: list[MoleculeNode] = []
        reactant_smiles_list: list[SmilesStr] = []

        for i, reactant_mol_input in enumerate(raw_reaction_node.children):
            reactant_node = build_tree_from_bipartite_node(
                raw_mol_node=reactant_mol_input, path_prefix=f"{path_prefix}-{i}"
            )
            reactants.append(reactant_node)
            reactant_smiles_list.append(reactant_node.smiles)

        reaction_smiles = ReactionSmilesStr(f"{'.'.join(sorted(reactant_smiles_list))}>>{canon_smiles}")

        reactions.append(
            ReactionNode(
                id=path_prefix.replace("ursa-mol", "ursa-rxn"),
                reaction_smiles=reaction_smiles,
                reactants=reactants,
            )
        )

    return MoleculeNode(
        id=path_prefix,
        molecule_hash=generate_molecule_hash(canon_smiles),
        smiles=canon_smiles,
        is_starting_material=is_starting_mat or not bool(reactions),
        # a molecule is a starting material if it's in stock, OR if it's not in stock but has no path to make it.
        reactions=reactions,
    )


# --- pattern b: precursor map recursor ---


def build_tree_from_precursor_map(
    smiles: SmilesStr,
    precursor_map: PrecursorMap,
    path_prefix: str = "ursa-mol-root",
    visited_path: set[SmilesStr] | None = None,
) -> MoleculeNode:
    """
    recursively builds a `MoleculeNode` from a precursor map, with cycle detection.
    this is the common logic for models like retro*, dreamretro, and processed ttlretro.
    """
    if visited_path is None:
        visited_path = set()

    if smiles in visited_path:
        logger.warning(f"cycle detected in route graph involving smiles: {smiles}. treating as a leaf node.")
        return MoleculeNode(
            id=path_prefix,
            molecule_hash=generate_molecule_hash(smiles),
            smiles=smiles,
            is_starting_material=True,
            reactions=[],
        )

    new_visited_path = visited_path | {smiles}
    is_starting_mat = smiles not in precursor_map
    reactions = []

    if not is_starting_mat:
        reactants: list[MoleculeNode] = []
        reactant_smiles_list = precursor_map[smiles]

        for i, reactant_smi in enumerate(reactant_smiles_list):
            reactant_node = build_tree_from_precursor_map(
                smiles=reactant_smi,
                precursor_map=precursor_map,
                path_prefix=f"{path_prefix}-{i}",
                visited_path=new_visited_path,
            )
            reactants.append(reactant_node)

        sorted_reactant_smiles = sorted(r.smiles for r in reactants)
        reaction_smiles = ReactionSmilesStr(f"{'.'.join(sorted_reactant_smiles)}>>{smiles}")

        reactions.append(
            ReactionNode(
                id=path_prefix.replace("ursa-mol", "ursa-rxn"),
                reaction_smiles=reaction_smiles,
                reactants=reactants,
            )
        )

    return MoleculeNode(
        id=path_prefix,
        molecule_hash=generate_molecule_hash(smiles),
        smiles=smiles,
        is_starting_material=is_starting_mat,
        reactions=reactions,
    )
