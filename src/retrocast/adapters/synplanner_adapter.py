from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_tree_from_bipartite_node
from retrocast.domain.schemas import BenchmarkTree, TargetInfo
from retrocast.exceptions import AdapterLogicError, UrsaException
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


# a discriminated union to handle the bipartite graph structure.
SynPlannerNode = Annotated[SynPlannerMoleculeInput | SynPlannerReactionInput, Field(discriminator="type")]


class SynPlannerRouteList(RootModel[list[SynPlannerMoleculeInput]]):
    """the top-level object for a single target is a list of potential routes."""

    pass


class SynPlannerAdapter(BaseAdapter):
    """adapter for converting synplanner-style outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates raw synplanner data, transforms it, and yields benchmarktree objects.
        """
        try:
            validated_routes = SynPlannerRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(
                f"  - raw data for target '{target_info.id}' failed synplanner schema validation. error: {e}"
            )
            return

        for synplanner_tree_root in validated_routes.root:
            try:
                tree = self._transform(synplanner_tree_root, target_info)
                yield tree
            except UrsaException as e:
                logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, synplanner_root: SynPlannerMoleculeInput, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single synplanner output tree.
        raises ursaexception on failure.
        """
        # refactor: use the common recursive builder.
        retrosynthetic_tree = build_tree_from_bipartite_node(raw_mol_node=synplanner_root, path_prefix="ursa-mol-root")

        if retrosynthetic_tree.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {retrosynthetic_tree.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)
