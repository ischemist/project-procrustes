from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_molecule_from_bipartite_node
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.schemas import Route, TargetInput
from retrocast.utils.logging import logger

# --- pydantic models for input validation ---
# these models validate the raw aizynthfinder output format before any transformation.


class AizynthBaseNode(BaseModel):
    """a base model for shared fields between node types."""

    smiles: str
    children: list[AizynthNode] = Field(default_factory=list)


class AizynthMoleculeInput(AizynthBaseNode):
    """represents a 'mol' node in the raw aizynth tree."""

    type: Literal["mol"]
    in_stock: bool = False


class AizynthReactionInput(AizynthBaseNode):
    """represents a 'reaction' node in the raw aizynth tree."""

    type: Literal["reaction"]
    metadata: dict[str, Any] = Field(default_factory=dict)


# a discriminated union to handle the bipartite graph structure.
AizynthNode = Annotated[AizynthMoleculeInput | AizynthReactionInput, Field(discriminator="type")]


class AizynthRouteList(RootModel[list[AizynthMoleculeInput]]):
    """the top-level object for a single target is a list of potential routes."""

    pass


class AizynthAdapter(BaseAdapter):
    """adapter for converting aizynthfinder-style outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInput) -> Generator[Route, None, None]:
        """
        validates raw aizynthfinder data, transforms it, and yields route objects.
        """
        try:
            validated_routes = AizynthRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target_info.id}' failed aizynth schema validation. error: {e}")
            return

        for rank, aizynth_tree_root in enumerate(validated_routes.root, start=1):
            try:
                route = self._transform(aizynth_tree_root, target_info, rank)
                yield route
            except RetroCastException as e:
                logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, aizynth_root: AizynthMoleculeInput, target_info: TargetInput, rank: int) -> Route:
        """
        orchestrates the transformation of a single aizynthfinder output tree.
        raises RetroCastException on failure.
        """
        # use the common recursive builder with new schema
        target_molecule = build_molecule_from_bipartite_node(raw_mol_node=aizynth_root)

        if target_molecule.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {target_molecule.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return Route(target=target_molecule, rank=rank, metadata={})
