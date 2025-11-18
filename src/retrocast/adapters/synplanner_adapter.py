from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.domain.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetInput
from retrocast.utils.logging import logger

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

    def cast(self, raw_target_data: Any, target_info: TargetInput) -> Generator[Route, None, None]:
        """
        validates raw synplanner data, transforms it, and yields route objects.
        """
        try:
            validated_routes = SynPlannerRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(
                f"  - raw data for target '{target_info.id}' failed synplanner schema validation. error: {e}"
            )
            return

        for rank, synplanner_tree_root in enumerate(validated_routes.root, start=1):
            try:
                route = self._transform(synplanner_tree_root, target_info, rank)
                yield route
            except RetroCastException as e:
                logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, synplanner_root: SynPlannerMoleculeInput, target_info: TargetInput, rank: int) -> Route:
        """
        orchestrates the transformation of a single synplanner output tree.
        raises RetroCastException on failure.
        """
        # use the custom recursive builder for synplanner (has mapped_smiles on reaction nodes)
        target_molecule = self._build_molecule_from_synplanner_node(synplanner_root)

        if target_molecule.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {target_molecule.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return Route(target=target_molecule, rank=rank, metadata={})

    def _build_molecule_from_synplanner_node(self, raw_mol_node: SynPlannerMoleculeInput) -> Molecule:
        """
        recursively builds a `Molecule` from a raw synplanner bipartite graph node.
        synplanner has mapped_smiles in the 'smiles' field of reaction nodes.
        """
        if raw_mol_node.type != "mol":
            raise AdapterLogicError(f"Expected node type 'mol' but got '{raw_mol_node.type}'")

        canon_smiles = canonicalize_smiles(raw_mol_node.smiles)
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

        raw_reaction_node: SynPlannerReactionInput = raw_mol_node.children[0]
        if raw_reaction_node.type != "reaction":
            raise AdapterLogicError("Child of molecule node was not a reaction node")

        # Build reactants recursively
        reactant_molecules: list[Molecule] = []
        for reactant_mol_input in raw_reaction_node.children:
            reactant_mol = self._build_molecule_from_synplanner_node(reactant_mol_input)
            reactant_molecules.append(reactant_mol)

        # Extract mapped_smiles from the 'smiles' field of the reaction node
        mapped_smiles = raw_reaction_node.smiles if hasattr(raw_reaction_node, "smiles") else None

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
