from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_node_type_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.adapters.paroutes_diagnostics import build_condition_slot_metadata
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import ReactionSmilesStr, SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---
# this format is effectively identical to aizynthfinder's output,
# just with different metadata in the reaction nodes.


class PaRoutesReactionMetadata(BaseModel):
    id: str = Field(..., alias="ID")
    rsmi: str | None = None  # reaction-mapped SMILES
    smiles: str | None = None  # mapped reactants>>product without the condition slot
    reaction_hash: str | None = None
    ring_breaker: bool | None = Field(None, alias="RingBreaker")


class PaRoutesBaseNode(BaseModel):
    smiles: str
    children: list[PaRoutesNode] = Field(default_factory=list)


class PaRoutesMoleculeInput(PaRoutesBaseNode):
    type: Literal["mol"]
    in_stock: bool = False


class PaRoutesReactionInput(PaRoutesBaseNode):
    type: Literal["reaction"]
    metadata: PaRoutesReactionMetadata
    children: list[PaRoutesMoleculeInput] = Field(default_factory=list)


PaRoutesNode = Annotated[PaRoutesMoleculeInput | PaRoutesReactionInput, Field(discriminator="type")]

# pydantic needs this to resolve the forward references in the recursive models
PaRoutesMoleculeInput.model_rebuild()
PaRoutesReactionInput.model_rebuild()


class PaRoutesAdapter(BaseAdapter):
    """adapter for converting paroutes experimental routes to the route schema."""

    def _get_patent_ids(self, node: PaRoutesMoleculeInput, visited: set[str] | None = None) -> set[str]:
        """
        recursively traverses the raw tree to collect all unique patent ids from reaction nodes.

        Args:
            node: The molecule node to traverse
            visited: Set of SMILES already visited (for cycle detection)

        Returns:
            Set of unique patent IDs found in the tree
        """
        if visited is None:
            visited = set()

        # Use raw SMILES for cycle detection (before canonicalization)
        if node.smiles in visited:
            logger.warning(f"cycle detected in _get_patent_ids for smiles: {node.smiles}")
            return set()

        new_visited = visited | {node.smiles}
        patent_ids: set[str] = set()

        for reaction_node in node.children:
            # Type guard: reaction nodes should have metadata
            if not isinstance(reaction_node, PaRoutesReactionInput):
                continue

            try:
                # the patent id is the part before the first semicolon
                patent_id = reaction_node.metadata.id.split(";")[0]
                patent_ids.add(patent_id)
            except (IndexError, AttributeError):
                logger.warning(f"could not parse patent id from metadata: {reaction_node.metadata}")

            for reactant_node in reaction_node.children:
                patent_ids.update(self._get_patent_ids(reactant_node, visited=new_visited))
        return patent_ids

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            validated_route_root = PaRoutesMoleculeInput.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("paroutes", target_id, "invalid molecule route root") from e

        yield RawRouteEntry(
            payload=validated_route_root,
            source_key=source_key,
            target_hint_id=None,
            target_hint_smiles=None,
            source_order=1,
        )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, PaRoutesMoleculeInput):
            raw_route = PaRoutesMoleculeInput.model_validate(raw_route)

        target_id = expected_target.id if expected_target is not None else "<unknown>"
        patent_ids = self._get_patent_ids(raw_route)
        if len(patent_ids) > 1:
            raise AdapterLogicError(
                f"PaRoutes route for target '{target_id}' contains reactions from multiple patents",
                code="adapter.multiple_patents",
                context={"adapter": "paroutes", "target_id": target_id, "patent_ids": sorted(patent_ids)},
            )
        elif len(patent_ids) == 1:
            patent_id = list(patent_ids)[0]

        if not patent_ids:
            raise AdapterLogicError(
                f"PaRoutes route for target '{target_id}' does not contain a patent id",
                code="adapter.patent_id_missing",
                context={"adapter": "paroutes", "target_id": target_id},
            )

        return self._transform(
            raw_route,
            expected_target,
            patent_id=patent_id,
            ignore_stereo=ignore_stereo,
        )

    def _transform(
        self,
        paroutes_root: PaRoutesMoleculeInput,
        target: TargetIdentity | None,
        patent_id: str,
        *,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        orchestrates the transformation of a single validated paroutes tree.
        """
        # build the molecule tree recursively with cycle detection
        target_molecule = self._build_molecule(
            paroutes_root,
            visited=set(),
            ignore_stereo=ignore_stereo,
        )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "paroutes",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        # add patent ID to route metadata (everything up to first semicolon)
        route_metadata = {"patent_id": patent_id}

        return Route(target=target_molecule, metadata=route_metadata)

    def _build_molecule(
        self,
        raw_mol_node: PaRoutesMoleculeInput,
        visited: set[SmilesStr] | None = None,
        *,
        ignore_stereo: bool = False,
    ) -> Molecule:
        """
        recursively builds a molecule from a paroutes bipartite graph node.

        Args:
            raw_mol_node: The raw molecule node from paroutes data
            visited: Set of canonical SMILES already visited (for cycle detection)
            ignore_stereo: If True, stereochemistry is stripped during SMILES canonicalization.

        Raises:
            AdapterLogicError: If a cycle is detected in the route graph
        """
        if raw_mol_node.type != "mol":
            raise adapter_node_type_error("paroutes", expected="mol", actual=raw_mol_node.type, role="molecule")

        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(raw_mol_node.smiles, ignore_stereo=ignore_stereo)

        # Cycle detection: check if we've seen this molecule before in the current path
        if canon_smiles in visited:
            raise adapter_cycle_error("paroutes", canon_smiles)

        # Create new visited set with current molecule added
        new_visited = visited | {canon_smiles}

        is_leaf = raw_mol_node.in_stock or not bool(raw_mol_node.children)

        if is_leaf:
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # in a valid tree, a molecule has at most one reaction leading to it
        if len(raw_mol_node.children) > 1:
            logger.warning(
                f"molecule {canon_smiles} has multiple child reactions in raw output; only the first is used in a tree."
            )

        first_child = raw_mol_node.children[0]
        if not isinstance(first_child, PaRoutesReactionInput):
            actual = getattr(first_child, "type", type(first_child).__name__)
            raise adapter_node_type_error("paroutes", expected="reaction", actual=actual, role="molecule child")
        raw_reaction_node: PaRoutesReactionInput = first_child

        # build reactants recursively with updated visited set
        reactant_molecules: list[Molecule] = []
        for reactant_mol_input in raw_reaction_node.children:
            reactant_mol = self._build_molecule(
                reactant_mol_input,
                visited=new_visited,
                ignore_stereo=ignore_stereo,
            )
            reactant_molecules.append(reactant_mol)

        # extract mapped smiles (rsmi) from metadata
        rxn_metadata = raw_reaction_node.metadata
        mapped_smiles_str = rxn_metadata.rsmi if rxn_metadata else None
        mapped_smiles = ReactionSmilesStr(mapped_smiles_str) if mapped_smiles_str else None
        metadata_dict = build_condition_slot_metadata(
            source_id=rxn_metadata.id,
            rsmi=rxn_metadata.rsmi,
            ring_breaker=rxn_metadata.ring_breaker,
            ignore_stereo=ignore_stereo,
        )
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=mapped_smiles,
            template=None,
            reagents=None,
            solvents=None,
            metadata=metadata_dict,
        )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )
