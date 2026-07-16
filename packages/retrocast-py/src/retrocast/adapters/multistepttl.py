from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.models.route import Molecule, Route
from retrocast.models.task import Target

# SECTION: Raw MultiStepTTL Schema


class TtlReaction(BaseModel):
    product: str
    reactants: list[str]


class TtlRoute(BaseModel):
    reactions: list[TtlReaction]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TtlRouteList(RootModel[list[TtlRoute]]):
    pass


# SECTION: Adapter


class MultiStepTTLAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            routes = TtlRouteList.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("multistepttl", target_id, "invalid pre-processed route list") from exc
        for source_order, route in enumerate(routes.root, start=1):
            yield RawRouteEntry(payload=route, source_key=source_key, source_order=source_order)

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        try:
            route = TtlRoute.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("multistepttl", target_id, "invalid route") from exc

        if not route.reactions:
            if target is None:
                raise AdapterLogicError(
                    "MultiStepTTL zero-reaction route needs a target",
                    code="adapter.route_transform_failed",
                    context={"adapter": "multistepttl", "target_id": target_id},
                )
            target_smiles = canonicalize_smiles(target.smiles)
            return Route(
                target=Molecule(smiles=target_smiles, inchikey=get_inchi_key(target_smiles)),
                annotations=route.metadata,
            )

        try:
            root_smiles = canonicalize_smiles(route.reactions[0].product)
        except InvalidSmilesError:
            if mode == "prune":
                raise AdapterLogicError(
                    "MultiStepTTL target molecule was pruned",
                    code="adapter.target_pruned",
                    context={"adapter": "multistepttl", "target_id": target_id},
                ) from None
            raise
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if root_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "multistepttl", target.id, expected_smiles=expected_smiles, actual_smiles=root_smiles
                )
        precursor_map = {}
        for reaction in route.reactions:
            try:
                product_smiles = canonicalize_smiles(reaction.product)
            except InvalidSmilesError:
                if mode == "strict":
                    raise
                continue
            precursor_map[product_smiles] = reaction.reactants
        route_target = build_molecule_from_precursor_map(root_smiles, precursor_map, adapter="multistepttl", mode=mode)
        if route_target is None:
            raise AdapterLogicError(
                "MultiStepTTL target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "multistepttl", "target_id": target_id},
            )
        return Route(target=route_target, annotations=route.metadata)
