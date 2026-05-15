from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_transform_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrochimeraRoutePayload:
    route: RetrochimeraRoute
    target_smiles: str


# --- pydantic models for input validation ---


class RetrochimeraReaction(BaseModel):
    reactants: list[str]
    product: str
    probability: float
    metadata: dict[str, Any] = {}


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


class RetrochimeraAdapter(BaseAdapter):
    """adapter for converting retrochimera-style outputs to the Route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw RetroChimera data and expose one route-like payload per route.
        """
        target_id = source_key or "<unknown>"
        try:
            validated_data = RetrochimeraData.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("retrochimera", target_id, "invalid output") from e

        if validated_data.result.error is not None:
            error_msg = validated_data.result.error.get("message", "unknown error")
            error_type = validated_data.result.error.get("type", "unknown")
            raise adapter_route_transform_error(
                "retrochimera",
                target_id,
                f"model reported {error_type}: {error_msg}",
                error_type=error_type,
            )

        if validated_data.result.outputs is None:
            raise adapter_route_transform_error(
                "retrochimera",
                target_id,
                "validated payload is missing result outputs",
                payload_field="result.outputs",
            )

        rank = 1
        for output in validated_data.result.outputs:
            for route in output.routes:
                yield RawRouteEntry(
                    payload=RetrochimeraRoutePayload(route=route, target_smiles=validated_data.smiles),
                    source_key=source_key,
                    target_hint_id=None,
                    target_hint_smiles=None,
                    source_order=rank,
                )
                rank += 1

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, RetrochimeraRoutePayload):
            raise adapter_schema_error(
                "retrochimera",
                expected_target.id if expected_target is not None else "<unknown>",
                "expected a retrochimera route payload",
            )
        return self._transform(
            raw_route.route,
            target_smiles=raw_route.target_smiles,
            target=expected_target,
            ignore_stereo=ignore_stereo,
        )

    def _transform(
        self,
        route: RetrochimeraRoute,
        target_smiles: str,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        orchestrates the transformation of a single retrochimera route.
        raises RetroCastException on failure.
        """
        precursor_map = self._build_precursor_map(route, ignore_stereo=ignore_stereo)
        actual_smiles = canonicalize_smiles(target_smiles, ignore_stereo=ignore_stereo)
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if actual_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "retrochimera",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=actual_smiles,
                )
            route_target_smiles = SmilesStr(target.smiles)
        else:
            route_target_smiles = actual_smiles

        target_molecule = build_molecule_from_precursor_map(
            smiles=route_target_smiles,
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter="retrochimera",
        )

        return Route(target=target_molecule, metadata={})

    def _build_precursor_map(
        self, route: RetrochimeraRoute, ignore_stereo: bool = False
    ) -> dict[SmilesStr, list[SmilesStr]]:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: dict[SmilesStr, list[SmilesStr]] = {}
        for reaction in route.reactions:
            canon_product = canonicalize_smiles(reaction.product, ignore_stereo=ignore_stereo)
            canon_reactants = [canonicalize_smiles(r, ignore_stereo=ignore_stereo) for r in reaction.reactants]
            precursor_map[canon_product] = canon_reactants
        return precursor_map
