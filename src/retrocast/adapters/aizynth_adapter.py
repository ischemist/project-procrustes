from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast._warnings import warn_deprecated
from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_bipartite_node
from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetIdentity

logger = logging.getLogger(__name__)

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
    scores: dict[str, float] = Field(default_factory=dict)


class AizynthReactionInput(AizynthBaseNode):
    """represents a 'reaction' node in the raw aizynth tree."""

    type: Literal["reaction"]
    metadata: dict[str, Any] = Field(default_factory=dict)


# a discriminated union to handle the bipartite graph structure.
AizynthNode = Annotated[AizynthMoleculeInput | AizynthReactionInput, Field(discriminator="type")]


class AizynthRouteList(RootModel[list[AizynthMoleculeInput]]):
    """the top-level object for a single target is a list of potential routes."""

    pass


class AiZynthFinderAdapter(BaseAdapter):
    """adapter for converting aizynthfinder-style outputs to the route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            validated_routes = AizynthRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("aizynth", target_id, "invalid route list") from e

        for rank, aizynth_tree_root in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=aizynth_tree_root,
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
        try:
            aizynth_root = AizynthMoleculeInput.model_validate(raw_route)
        except ValidationError as e:
            raise adapter_schema_error(
                "aizynth",
                expected_target.id if expected_target is not None else "<unknown>",
                "invalid molecule route root",
            ) from e
        target_molecule = build_molecule_from_bipartite_node(
            raw_mol_node=aizynth_root,
            ignore_stereo=ignore_stereo,
            adapter="aizynth",
        )

        if expected_target is not None:
            expected_smiles = canonicalize_smiles(expected_target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "aizynth",
                    expected_target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        route_metadata: dict[str, Any] = {}
        if aizynth_root.scores:
            route_metadata["scores"] = aizynth_root.scores
            state_score = aizynth_root.scores.get("state score")
            if state_score is not None:
                route_metadata["state_score"] = state_score

        return Route(target=target_molecule, metadata=route_metadata)


_DEPRECATED_ADAPTER_ALIASES: dict[str, type[BaseAdapter]] = {
    "AizynthAdapter": AiZynthFinderAdapter,
}


def __getattr__(name: str) -> Any:
    adapter_type = _DEPRECATED_ADAPTER_ALIASES.get(name)
    if adapter_type is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        old=f"{__name__}.{name}",
        new=f"{__name__}.AiZynthFinderAdapter",
        remove_in="0.7",
        stacklevel=2,
    )
    globals()[name] = adapter_type
    return adapter_type
