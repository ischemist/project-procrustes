from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import BenchmarkTree, MoleculeNode, ReactionNode, TargetInfo
from retrocast.exceptions import AdapterLogicError, UrsaException
from retrocast.typing import ReactionSmilesStr, SmilesStr
from retrocast.utils.hashing import generate_molecule_hash
from retrocast.utils.logging import logger


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
    """Adapter for converting DMS-style model outputs to the BenchmarkTree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        Validates raw DMS data, transforms it, and yields BenchmarkTree objects.
        """
        try:
            # 1. Model-specific validation happens HERE, inside the adapter.
            validated_routes = DMSRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - Raw data for target '{target_info.id}' failed DMS schema validation. Error: {e}")
            return  # Stop processing this target

        # 2. Iterate and transform each valid route
        for dms_tree_root in validated_routes.root:
            try:
                # The private _transform method now only handles one route at a time
                tree = self._transform(dms_tree_root, target_info)
                yield tree
            except UrsaException as e:
                # A single route failed, log it and continue with the next one.
                logger.warning(f"  - Route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, raw_data: DMSTree, target_info: TargetInfo) -> BenchmarkTree:
        """
        Orchestrates the transformation of a single DMS output tree.
        Raises UrsaException on failure.
        """
        # begin the recursion from the root node
        retrosynthetic_tree = self._build_molecule_node(dms_node=raw_data, path_prefix="ursa-mol-root")

        # Final validation: does the transformed tree root match the canonical target smiles?
        if retrosynthetic_tree.smiles != target_info.smiles:
            # this is a logic error, not a parse error
            msg = (
                f"Mismatched SMILES for target {target_info.id}. "
                f"Expected canonical: {target_info.smiles}, but adapter produced: {retrosynthetic_tree.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _build_molecule_node(
        self, dms_node: DMSTree, path_prefix: str, visited_path: set[SmilesStr] | None = None
    ) -> MoleculeNode:
        """
        Recursively builds a MoleculeNode. This will propagate InvalidSmilesError if it occurs.
        """
        if visited_path is None:
            visited_path = set()

        canon_smiles = canonicalize_smiles(dms_node.smiles)

        if canon_smiles in visited_path:
            raise AdapterLogicError(f"cycle detected in route graph involving smiles: {canon_smiles}")

        new_visited_path = visited_path | {canon_smiles}
        is_starting_mat = not bool(dms_node.children)
        reactions = []

        if not is_starting_mat:
            reactants: list[MoleculeNode] = []
            reactant_smiles_list: list[SmilesStr] = []

            for i, child_node in enumerate(dms_node.children):
                reactant_node = self._build_molecule_node(
                    dms_node=child_node, path_prefix=f"{path_prefix}-{i}", visited_path=new_visited_path
                )
                reactants.append(reactant_node)
                reactant_smiles_list.append(reactant_node.smiles)

            reaction_smiles = ReactionSmilesStr(f"{'.'.join(sorted(reactant_smiles_list))}>>{canon_smiles}")

            reactions.append(
                ReactionNode(
                    id=path_prefix.replace("ursa-mol", "ursa-rxn"), reaction_smiles=reaction_smiles, reactants=reactants
                )
            )

        return MoleculeNode(
            id=path_prefix,
            molecule_hash=generate_molecule_hash(canon_smiles),
            smiles=canon_smiles,
            is_starting_material=is_starting_mat,
            reactions=reactions,
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
