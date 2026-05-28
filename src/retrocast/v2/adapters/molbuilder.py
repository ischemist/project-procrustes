from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.adapters.common import build_plain_tree_molecule
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target

# SECTION: Raw MolBuilder Schema


class MolBuilderPrecursor(BaseModel):
    smiles: str
    name: str = ""
    cost_per_kg: float = 0.0


class MolBuilderDisconnection(BaseModel):
    reaction_name: str
    named_reaction: str | None = None
    category: str = ""
    score: float = 0.0
    precursors: list[MolBuilderPrecursor] = Field(default_factory=list)


class MolBuilderNode(BaseModel):
    smiles: str
    is_purchasable: bool = False
    functional_groups: list[str] = Field(default_factory=list)
    best_disconnection: MolBuilderDisconnection | None = None
    children: list[MolBuilderNode] = Field(default_factory=list)


class MolBuilderRouteList(RootModel[list[MolBuilderNode]]):
    pass


# SECTION: Adapter


class MolBuilderAdapter:
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            routes = MolBuilderRouteList.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("molbuilder", target_id, "invalid route list") from exc

        for source_order, route_root in enumerate(routes.root, start=1):
            yield RawRouteEntry(payload=route_root, source_key=source_key, source_order=source_order)

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        try:
            route_root = MolBuilderNode.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("molbuilder", target_id, "invalid route root") from exc

        route_target = build_plain_tree_molecule(
            route_root,
            adapter="molbuilder",
            mode=mode,
            get_smiles=lambda node: node.smiles,
            get_children=lambda node: [] if node.is_purchasable else node.children,
            get_molecule_annotations=_molecule_annotations,
            get_reaction_fields=_reaction_fields,
        )
        if route_target is None:
            raise AdapterLogicError(
                "MolBuilder target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "molbuilder", "target_id": target_id},
            )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "molbuilder",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=route_target.smiles,
                )

        return Route(target=route_target)


# SECTION: Helpers


def _molecule_annotations(node: MolBuilderNode) -> dict[str, Any]:
    if not node.functional_groups:
        return {}
    return {"functional_groups": node.functional_groups}


def _reaction_fields(node: MolBuilderNode) -> dict[str, Any]:
    disconnection = node.best_disconnection
    if disconnection is None:
        return {}

    annotations: dict[str, Any] = {"score": disconnection.score}
    reaction_name = disconnection.reaction_name.strip()
    if reaction_name:
        annotations["reaction_name"] = reaction_name
    if disconnection.named_reaction:
        annotations["named_reaction"] = disconnection.named_reaction
    if disconnection.category:
        annotations["category"] = disconnection.category
    if disconnection.precursors:
        annotations["precursors"] = [precursor.model_dump() for precursor in disconnection.precursors]

    return {"template": reaction_name or None, "annotations": annotations}
