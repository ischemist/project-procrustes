from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.errors import (
    adapter_route_transform_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.adapters.common import build_molecule_from_precursor_map
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target

# SECTION: Raw RetroChimera Schema


@dataclass(frozen=True, slots=True)
class RetrochimeraRoutePayload:
    route: RetrochimeraRoute
    target_smiles: str
    annotations: tuple[tuple[str, Any], ...]

    def route_annotations(self) -> dict[str, Any]:
        return dict(self.annotations)


class RetrochimeraReaction(BaseModel):
    reactants: list[str]
    product: str
    probability: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrochimeraRoute(BaseModel):
    reactions: list[RetrochimeraReaction]
    num_steps: int
    step_probability_min: float
    step_probability_product: float


class RetrochimeraOutput(BaseModel):
    routes: list[RetrochimeraRoute]
    num_routes: int
    num_routes_initial_extraction: int = 0
    target_is_purchasable: bool = False
    num_model_calls_total: int = 0
    num_model_calls_new: int = 0
    num_model_calls_cached: int = 0
    num_nodes_explored: int = 0
    time_taken_s_search: float = 0.0
    time_taken_s_extraction: float = 0.0


class RetrochimeraResult(BaseModel):
    request: dict[str, Any] | None = None
    outputs: list[RetrochimeraOutput] | None = None
    error: dict[str, Any] | None = None
    time_taken_s: float = 0.0


class RetrochimeraData(BaseModel):
    smiles: str
    result: RetrochimeraResult


# SECTION: Adapter


class RetroChimeraAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            data = RetrochimeraData.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("retrochimera", target_id, "invalid output") from exc
        if data.result.error is not None:
            error_type = data.result.error.get("type", "unknown")
            error_msg = data.result.error.get("message", "unknown error")
            raise adapter_route_transform_error("retrochimera", target_id, f"model reported {error_type}: {error_msg}")
        if data.result.outputs is None:
            raise adapter_route_transform_error(
                "retrochimera", target_id, "validated payload is missing result outputs"
            )

        source_order = 1
        for output in data.result.outputs:
            annotations = output.model_dump(exclude={"routes"})
            for route in output.routes:
                yield RawRouteEntry(
                    payload=RetrochimeraRoutePayload(
                        route=route.model_copy(deep=True),
                        target_smiles=data.smiles,
                        annotations=tuple((key, deepcopy(value)) for key, value in annotations.items()),
                    ),
                    source_key=source_key,
                    source_order=source_order,
                )
                source_order += 1

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        if not isinstance(raw_route, RetrochimeraRoutePayload):
            raise adapter_schema_error("retrochimera", target_id, "expected a retrochimera route payload")
        root_smiles = canonicalize_smiles(raw_route.target_smiles)
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if root_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "retrochimera",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=root_smiles,
                )
        precursor_map = {
            canonicalize_smiles(reaction.product): reaction.reactants for reaction in raw_route.route.reactions
        }
        reaction_annotations = {
            canonicalize_smiles(reaction.product): {"probability": reaction.probability, **reaction.metadata}
            for reaction in raw_route.route.reactions
        }
        route_target = build_molecule_from_precursor_map(
            root_smiles,
            precursor_map,
            adapter="retrochimera",
            mode=mode,
            reaction_annotations=reaction_annotations,
        )
        if route_target is None:
            raise AdapterLogicError(
                "RetroChimera target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "retrochimera", "target_id": target_id},
            )
        return Route(target=route_target, annotations=raw_route.route_annotations())
