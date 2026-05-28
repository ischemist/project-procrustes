from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.adapters.common import build_bipartite_molecule
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target

# SECTION: Raw Syntheseus Schema


class SyntheseusBaseNode(BaseModel):
    smiles: str
    children: list[SyntheseusNode] = Field(default_factory=list)


class SyntheseusMoleculeInput(SyntheseusBaseNode):
    type: Literal["mol"]
    in_stock: bool = False


class SyntheseusReactionInput(SyntheseusBaseNode):
    type: Literal["reaction"]
    metadata: dict[str, Any] = Field(default_factory=dict)


SyntheseusNode = Annotated[SyntheseusMoleculeInput | SyntheseusReactionInput, Field(discriminator="type")]


class SyntheseusRouteList(RootModel[list[SyntheseusMoleculeInput]]):
    pass


# SECTION: Adapter


class SyntheseusAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            routes = SyntheseusRouteList.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("syntheseus", target_id, "invalid route list") from exc
        for source_order, route_root in enumerate(routes.root, start=1):
            yield RawRouteEntry(payload=route_root, source_key=source_key, source_order=source_order)

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        try:
            route_root = SyntheseusMoleculeInput.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("syntheseus", target_id, "invalid molecule route root") from exc
        route_target = build_bipartite_molecule(
            route_root,
            adapter="syntheseus",
            mode=mode,
            reaction_fields=_reaction_fields,
        )
        if route_target is None:
            raise AdapterLogicError(
                "Syntheseus target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "syntheseus", "target_id": target_id},
            )
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "syntheseus", target.id, expected_smiles=expected_smiles, actual_smiles=route_target.smiles
                )
        return Route(target=route_target)


# SECTION: Helpers


def _reaction_fields(node: SyntheseusReactionInput) -> dict[str, Any]:
    fields: dict[str, Any] = {"annotations": dict(node.metadata)}
    if "mapped_reaction_smiles" in node.metadata:
        fields["mapped_reaction_smiles"] = node.metadata["mapped_reaction_smiles"]
    if "template" in node.metadata:
        fields["template"] = node.metadata["template"]
    return fields
