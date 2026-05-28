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

# SECTION: Raw DMS Schema


class DMSTree(BaseModel):
    smiles: str
    children: list[DMSTree] = Field(default_factory=list)


class DMSRouteList(RootModel[list[DMSTree]]):
    pass


# SECTION: Adapter


class DirectMultiStepAdapter:
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            routes = DMSRouteList.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("dms", target_id, "invalid route list") from exc

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
            route_root = DMSTree.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("dms", target_id, "invalid route root") from exc

        route_target = build_plain_tree_molecule(
            route_root,
            adapter="dms",
            mode=mode,
            get_smiles=lambda node: node.smiles,
            get_children=lambda node: node.children,
        )
        if route_target is None:
            raise AdapterLogicError(
                "DMS target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "dms", "target_id": target_id},
            )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "dms",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=route_target.smiles,
                )
        return Route(target=route_target)

    @staticmethod
    def calculate_route_length(dms_node: DMSTree) -> int:
        if not dms_node.children:
            return 0
        return max(DirectMultiStepAdapter.calculate_route_length(child) for child in dms_node.children) + 1
