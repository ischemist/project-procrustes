from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ursa.adapters.base_adapter import BaseAdapter
from ursa.adapters.common import build_tree_from_bipartite_node
from ursa.domain.schemas import BenchmarkTree, TargetInfo
from ursa.exceptions import AdapterLogicError, UrsaException
from ursa.utils.logging import logger

# --- pydantic models for input validation ---
# this format is effectively identical to aizynthfinder's output,
# just with different metadata in the reaction nodes.


class PaRoutesReactionMetadata(BaseModel):
    id: str = Field(..., alias="ID")


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
    """adapter for converting paroutes experimental routes to the benchmarktree schema."""

    def _get_patent_ids(self, node: PaRoutesMoleculeInput) -> set[str]:
        """recursively traverses the raw tree to collect all unique patent ids from reaction nodes."""
        patent_ids: set[str] = set()
        for reaction_node in node.children:
            try:
                # the patent id is the part before the first semicolon
                patent_id = reaction_node.metadata.id.split(";")[0]
                patent_ids.add(patent_id)
            except (IndexError, AttributeError):
                logger.warning(f"could not parse patent id from metadata: {reaction_node.metadata}")

            for reactant_node in reaction_node.children:
                patent_ids.update(self._get_patent_ids(reactant_node))
        return patent_ids

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates a single paroutes route, checks for patent consistency, and transforms it.
        """
        try:
            # unlike other adapters, the raw data for one target is a single route object, not a list.
            validated_route_root = PaRoutesMoleculeInput.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target_info.id}' failed paroutes schema validation. error: {e}")
            return

        # --- custom validation: ensure all reactions are from the same patent ---
        patent_ids = self._get_patent_ids(validated_route_root)
        if len(patent_ids) > 1:
            logger.warning(
                f"  - skipping route for '{target_info.id}': contains reactions from multiple patents: {patent_ids}"
            )
            return

        try:
            tree = self._transform(validated_route_root, target_info)
            yield tree
        except UrsaException as e:
            logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
            return

    def _transform(self, paroutes_root: PaRoutesMoleculeInput, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single validated paroutes tree.
        """
        # this format is a standard bipartite graph, so we reuse the common builder.
        retrosynthetic_tree = build_tree_from_bipartite_node(raw_mol_node=paroutes_root, path_prefix="ursa-mol-root")

        if retrosynthetic_tree.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {retrosynthetic_tree.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)
