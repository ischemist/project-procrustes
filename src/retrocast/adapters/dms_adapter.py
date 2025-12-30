import logging
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


class DMSTree(BaseModel):
    """
    A Pydantic model for the raw output from "DMS" models.

    This recursively validates the structure of a synthetic tree node,
    ensuring it has a 'smiles' string and a list of 'children' nodes.
    """

    smiles: str  # we don't canonicalize yet; this is raw input
    children: list["DMSTree"] = Field(default_factory=list)


class DMSRouteList(RootModel[list[DMSTree]]):
    """
    Represents the raw model output for a single target, which is a list of routes.
    """

    pass


class DMSAdapter(BaseAdapter):
    """Adapter for converting DMS-style model outputs to the Route schema."""

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        Validates raw DMS data, transforms it, and yields Route objects.
        """
        try:
            # 1. Model-specific validation happens HERE, inside the adapter.
            validated_routes = DMSRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.debug(f"  - Raw data for target '{target.id}' failed DMS schema validation. Error: {e}")
            return  # Stop processing this target

        # 2. Iterate and transform each valid route
        for rank, dms_tree_root in enumerate(validated_routes.root, start=1):
            try:
                # The private _transform method now only handles one route at a time
                route = self._transform(dms_tree_root, target, rank, ignore_stereo=ignore_stereo)
                yield route
            except RetroCastException as e:
                # A single route failed, log it and continue with the next one.
                logger.debug(f"  - Route for '{target.id}' failed transformation: {e}")
                continue

    def _transform(self, raw_data: DMSTree, target: TargetIdentity, rank: int, ignore_stereo: bool = False) -> Route:
        """
        Orchestrates the transformation of a single DMS output tree.
        Raises RetroCastException on failure.
        """
        # Begin the recursion from the root node
        target_molecule = self._build_molecule(dms_node=raw_data, ignore_stereo=ignore_stereo)

        # Final validation: does the transformed tree root match the canonical target smiles?
        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if target_molecule.smiles != expected_smiles:
            # This is a logic error, not a parse error
            msg = (
                f"Mismatched SMILES for target {target.id}. "
                f"Expected canonical: {expected_smiles}, but adapter produced: {target_molecule.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return Route(target=target_molecule, rank=rank, metadata={})

    def _build_molecule(
        self, dms_node: DMSTree, visited: set[SmilesStr] | None = None, ignore_stereo: bool = False
    ) -> Molecule:
        """
        Recursively builds a Molecule from a DMS tree node.
        This will propagate InvalidSmilesError if it occurs.
        """
        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(dms_node.smiles, ignore_stereo=ignore_stereo)

        if canon_smiles in visited:
            raise AdapterLogicError(f"cycle detected in route graph involving smiles: {canon_smiles}")

        new_visited = visited | {canon_smiles}
        is_leaf = not bool(dms_node.children)

        if is_leaf:
            # This is a starting material (leaf node)
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # Build reactants recursively
        reactant_molecules: list[Molecule] = []
        for child_node in dms_node.children:
            reactant_mol = self._build_molecule(dms_node=child_node, visited=new_visited, ignore_stereo=ignore_stereo)
            reactant_molecules.append(reactant_mol)

        # Create the reaction step
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=None,
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

    @staticmethod
    def calculate_route_length(dms_node: DMSTree) -> int:
        """
        Calculate the length of a route from the raw DMS tree structure.

        This counts the number of reactions (steps) in the longest path
        from the target to any starting material.
        """
        if not dms_node.children:
            return 0

        max_child_length = 0
        for child in dms_node.children:
            child_length = DMSAdapter.calculate_route_length(child)
            max_child_length = max(max_child_length, child_length)

        return max_child_length + 1
