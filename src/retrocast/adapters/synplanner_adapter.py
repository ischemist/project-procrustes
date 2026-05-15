from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_node_type_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import ReactionSmilesStr, SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---
# these models validate the raw synplanner output format before any transformation.
# they are structurally identical to aizynthfinder's output.


class SynPlannerBaseNode(BaseModel):
    """a base model for shared fields between node types."""

    smiles: str
    children: list[SynPlannerNode] = Field(default_factory=list)


class SynPlannerMoleculeInput(SynPlannerBaseNode):
    """represents a 'mol' node in the raw synplanner tree."""

    type: Literal["mol"]
    in_stock: bool = False


class SynPlannerReactionInput(SynPlannerBaseNode):
    """represents a 'reaction' node in the raw synplanner tree."""

    type: Literal["reaction"]
    # synplanner has mapped_smiles in the 'smiles' field of reaction nodes


# a discriminated union to handle the bipartite graph structure.
SynPlannerNode = Annotated[SynPlannerMoleculeInput | SynPlannerReactionInput, Field(discriminator="type")]


class SynPlannerRouteList(RootModel[list[SynPlannerMoleculeInput]]):
    """the top-level object for a single target is a list of potential routes."""

    pass


class SynPlannerAdapter(BaseAdapter):
    """adapter for converting synplanner-style outputs to the route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
        expected_target: TargetIdentity | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw SynPlanner data and expose one route-like payload per entry.
        """
        target_id = expected_target.id if expected_target is not None else source_key or "<unknown>"
        try:
            validated_routes = SynPlannerRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("synplanner", target_id, "invalid route list") from e

        for rank, synplanner_tree_root in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=synplanner_tree_root,
                source_key=source_key,
                expected_target_id=expected_target.id if expected_target is not None else None,
                expected_target_smiles=expected_target.smiles if expected_target is not None else None,
                source_order=rank,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, SynPlannerMoleculeInput):
            raw_route = SynPlannerMoleculeInput.model_validate(raw_route)
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(
        self,
        synplanner_root: SynPlannerMoleculeInput,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        orchestrates the transformation of a single synplanner output tree.
        raises RetroCastException on failure.
        """
        # use the custom recursive builder for synplanner (has mapped_smiles on reaction nodes)
        target_molecule = self._build_molecule_from_synplanner_node(
            synplanner_root,
            ignore_stereo=ignore_stereo,
            visited=set(),
        )

        # canonicalize both synplanner output and benchmark target with RDKit to align formats
        produced = canonicalize_smiles(target_molecule.smiles, remove_mapping=True, ignore_stereo=ignore_stereo)
        if target is not None:
            expected = canonicalize_smiles(target.smiles, remove_mapping=True, ignore_stereo=ignore_stereo)

            if produced != expected:
                raise adapter_target_mismatch(
                    "synplanner",
                    target.id,
                    expected_smiles=expected,
                    actual_smiles=produced,
                )

            target_molecule = Molecule(
                smiles=target.smiles,
                inchikey=get_inchi_key(target.smiles),
                synthesis_step=target_molecule.synthesis_step,
                metadata=target_molecule.metadata,
            )

        return Route(target=target_molecule, metadata={})

    def _build_molecule_from_synplanner_node(
        self,
        raw_mol_node: SynPlannerMoleculeInput,
        ignore_stereo: bool = False,
        visited: set[SmilesStr] | None = None,
    ) -> Molecule:
        """
        recursively builds a `Molecule` from a raw synplanner bipartite graph node.
        synplanner has mapped_smiles in the 'smiles' field of reaction nodes.
        """
        if raw_mol_node.type != "mol":
            raise adapter_node_type_error("synplanner", expected="mol", actual=raw_mol_node.type, role="molecule")

        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(raw_mol_node.smiles, remove_mapping=True, ignore_stereo=ignore_stereo)
        if canon_smiles in visited:
            raise adapter_cycle_error("synplanner", canon_smiles)

        new_visited = visited | {canon_smiles}
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

        first_child = raw_mol_node.children[0]
        if not isinstance(first_child, SynPlannerReactionInput):
            actual = getattr(first_child, "type", type(first_child).__name__)
            raise adapter_node_type_error("synplanner", expected="reaction", actual=actual, role="molecule child")
        raw_reaction_node: SynPlannerReactionInput = first_child

        # Build reactants recursively
        reactant_molecules: list[Molecule] = []
        for reactant_mol_input in raw_reaction_node.children:
            # Type guard: children of reaction nodes should be molecule nodes
            if not isinstance(reactant_mol_input, SynPlannerMoleculeInput):
                actual = getattr(reactant_mol_input, "type", type(reactant_mol_input).__name__)
                raise adapter_node_type_error("synplanner", expected="mol", actual=actual, role="reaction child")
            reactant_mol = self._build_molecule_from_synplanner_node(
                reactant_mol_input,
                ignore_stereo=ignore_stereo,
                visited=new_visited,
            )
            reactant_molecules.append(reactant_mol)

        # Extract mapped_smiles from the 'smiles' field of the reaction node
        mapped_smiles = ReactionSmilesStr(raw_reaction_node.smiles) if hasattr(raw_reaction_node, "smiles") else None

        # Create the reaction step
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=mapped_smiles,
            reagents=None,
            solvents=None,
            metadata={},
        )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )
