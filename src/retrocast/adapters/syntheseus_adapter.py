from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_bipartite_node
from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetIdentity

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---
# these models validate the serialized syntheseus output.
# the structure is intentionally made identical to aizynthfinder's output.


class SyntheseusBaseNode(BaseModel):
    """a base model for shared fields between node types."""

    smiles: str
    children: list[SyntheseusNode] = Field(default_factory=list)


class SyntheseusMoleculeInput(SyntheseusBaseNode):
    """represents a 'mol' node in the raw tree."""

    type: Literal["mol"]
    in_stock: bool = False


class SyntheseusReactionInput(SyntheseusBaseNode):
    """represents a 'reaction' node in the raw tree."""

    type: Literal["reaction"]
    metadata: dict[str, Any] = Field(default_factory=dict)


# a discriminated union to handle the bipartite graph structure.
SyntheseusNode = Annotated[SyntheseusMoleculeInput | SyntheseusReactionInput, Field(discriminator="type")]


class SyntheseusRouteList(RootModel[list[SyntheseusMoleculeInput]]):
    """the top-level object for a single target is a list of potential routes."""

    pass


class SyntheseusAdapter(BaseAdapter):
    """adapter for converting serialized syntheseus outputs to the route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw Syntheseus data and expose one route-like payload per entry.
        """
        target_id = source_key or "<unknown>"
        try:
            validated_routes = SyntheseusRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("syntheseus", target_id, "invalid route list") from e

        for rank, syntheseus_tree_root in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=syntheseus_tree_root,
                source_key=source_key,
                target_hint_id=None,
                target_hint_smiles=None,
                source_order=rank,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, SyntheseusMoleculeInput):
            raw_route = SyntheseusMoleculeInput.model_validate(raw_route)
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(
        self,
        syntheseus_root: SyntheseusMoleculeInput,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        """
        orchestrates the transformation of a single serialized syntheseus output tree.
        raises RetroCastException on failure.
        """
        # use the common recursive builder with new schema
        target_molecule = build_molecule_from_bipartite_node(
            raw_mol_node=syntheseus_root,
            ignore_stereo=ignore_stereo,
            adapter="syntheseus",
        )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "syntheseus",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        return Route(target=target_molecule, metadata={})
