from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from retrocast.adapters.errors import adapter_cycle_error, adapter_node_type_error
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.base import AdaptMode
from retrocast.v2.models.route import Molecule, Reaction

ReactionAnnotations = Mapping[SmilesStr, Mapping[str, Any]]
ReactionFields = Mapping[str, Any]
MoleculeAnnotations = Mapping[str, Any]


# SECTION: Precursor Map Traversal


def build_molecule_from_precursor_map(
    smiles: str,
    precursor_map: Mapping[SmilesStr, Sequence[str]],
    *,
    adapter: str,
    mode: AdaptMode,
    reaction_annotations: ReactionAnnotations | None = None,
) -> Molecule | None:
    return _build_molecule_from_precursor_map(
        smiles,
        precursor_map,
        adapter=adapter,
        mode=mode,
        visited=set(),
        reaction_annotations=reaction_annotations,
    )


def _build_molecule_from_precursor_map(
    smiles: str,
    precursor_map: Mapping[SmilesStr, Sequence[str]],
    *,
    adapter: str,
    mode: AdaptMode,
    visited: set[SmilesStr],
    reaction_annotations: ReactionAnnotations | None,
) -> Molecule | None:
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
        reactant_molecule = _build_molecule_from_precursor_map(
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


# SECTION: Bipartite Molecule-Reaction Traversal


def build_bipartite_molecule(
    node: Any,
    *,
    adapter: str,
    mode: AdaptMode,
    reaction_fields: Callable[[Any], ReactionFields] | None = None,
    remove_mapping: bool = False,
) -> Molecule | None:
    return _build_bipartite_molecule(
        node,
        adapter=adapter,
        mode=mode,
        reaction_fields=reaction_fields,
        remove_mapping=remove_mapping,
        visited=set(),
    )


def _build_bipartite_molecule(
    node: Any,
    *,
    adapter: str,
    mode: AdaptMode,
    reaction_fields: Callable[[Any], ReactionFields] | None,
    remove_mapping: bool,
    visited: set[SmilesStr],
) -> Molecule | None:
    if getattr(node, "type", None) != "mol":
        actual = getattr(node, "type", type(node).__name__)
        raise adapter_node_type_error(adapter, expected="mol", actual=actual, role="molecule")

    try:
        canon_smiles = canonicalize_smiles(node.smiles, remove_mapping=remove_mapping)
    except InvalidSmilesError:
        if mode == "prune":
            return None
        raise

    if canon_smiles in visited:
        raise adapter_cycle_error(adapter, canon_smiles)

    if getattr(node, "in_stock", False) or not node.children:
        return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))

    if len(node.children) > 1:
        raise AdapterLogicError(
            f"{adapter} route is not a tree: molecule has multiple child reactions",
            code="adapter.route_not_tree",
            context={"adapter": adapter, "smiles": canon_smiles, "child_reaction_count": len(node.children)},
        )

    reaction_node = node.children[0]
    if getattr(reaction_node, "type", None) != "reaction":
        actual = getattr(reaction_node, "type", type(reaction_node).__name__)
        raise adapter_node_type_error(adapter, expected="reaction", actual=actual, role="molecule child")

    reactants: list[Molecule] = []
    for child in reaction_node.children:
        if getattr(child, "type", None) != "mol":
            actual = getattr(child, "type", type(child).__name__)
            raise adapter_node_type_error(adapter, expected="mol", actual=actual, role="reaction child")
        reactant = _build_bipartite_molecule(
            child,
            adapter=adapter,
            mode=mode,
            reaction_fields=reaction_fields,
            remove_mapping=remove_mapping,
            visited=visited | {canon_smiles},
        )
        if reactant is not None:
            reactants.append(reactant)

    if not reactants:
        if mode == "prune":
            return None
        raise AdapterLogicError(
            f"{adapter} reaction has no reactants",
            code="adapter.reaction_empty",
            context={"adapter": adapter, "smiles": canon_smiles},
        )

    fields = dict(reaction_fields(reaction_node)) if reaction_fields is not None else {}
    return Molecule(
        smiles=canon_smiles,
        inchikey=get_inchi_key(canon_smiles),
        product_of=Reaction(reactants=reactants, **fields),
    )


# SECTION: Plain Molecule Tree Traversal


def build_plain_tree_molecule(
    node: Any,
    *,
    adapter: str,
    mode: AdaptMode,
    get_smiles: Callable[[Any], str],
    get_children: Callable[[Any], Sequence[Any]],
    get_molecule_annotations: Callable[[Any], MoleculeAnnotations] | None = None,
    get_reaction_fields: Callable[[Any], ReactionFields] | None = None,
) -> Molecule | None:
    return _build_plain_tree_molecule(
        node,
        adapter=adapter,
        mode=mode,
        get_smiles=get_smiles,
        get_children=get_children,
        get_molecule_annotations=get_molecule_annotations,
        get_reaction_fields=get_reaction_fields,
        visited=set(),
    )


def _build_plain_tree_molecule(
    node: Any,
    *,
    adapter: str,
    mode: AdaptMode,
    get_smiles: Callable[[Any], str],
    get_children: Callable[[Any], Sequence[Any]],
    get_molecule_annotations: Callable[[Any], MoleculeAnnotations] | None,
    get_reaction_fields: Callable[[Any], ReactionFields] | None,
    visited: set[SmilesStr],
) -> Molecule | None:
    try:
        canon_smiles = canonicalize_smiles(get_smiles(node))
    except InvalidSmilesError:
        if mode == "prune":
            return None
        raise

    if canon_smiles in visited:
        raise adapter_cycle_error(adapter, canon_smiles)

    annotations = dict(get_molecule_annotations(node)) if get_molecule_annotations is not None else {}
    children = list(get_children(node))
    if not children:
        return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles), annotations=annotations)

    reactants: list[Molecule] = []
    for child in children:
        reactant = _build_plain_tree_molecule(
            child,
            adapter=adapter,
            mode=mode,
            get_smiles=get_smiles,
            get_children=get_children,
            get_molecule_annotations=get_molecule_annotations,
            get_reaction_fields=get_reaction_fields,
            visited=visited | {canon_smiles},
        )
        if reactant is not None:
            reactants.append(reactant)

    if not reactants:
        if mode == "prune":
            return None
        raise AdapterLogicError(
            f"{adapter} reaction has no reactants",
            code="adapter.reaction_empty",
            context={"adapter": adapter, "smiles": canon_smiles},
        )

    fields = dict(get_reaction_fields(node)) if get_reaction_fields is not None else {}
    return Molecule(
        smiles=canon_smiles,
        inchikey=get_inchi_key(canon_smiles),
        product_of=Reaction(reactants=reactants, **fields),
        annotations=annotations,
    )
