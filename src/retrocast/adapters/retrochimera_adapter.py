from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_transform_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

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

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        validates raw retrochimera data, transforms it, and yields Route objects.
        """
        try:
            validated_data = RetrochimeraData.model_validate(raw_target_data)
        except ValidationError as e:
            raise adapter_schema_error("retrochimera", target.id, "invalid output") from e

        if validated_data.result.error is not None:
            error_msg = validated_data.result.error.get("message", "unknown error")
            error_type = validated_data.result.error.get("type", "unknown")
            raise adapter_route_transform_error(
                "retrochimera",
                target.id,
                f"model reported {error_type}: {error_msg}",
                error_type=error_type,
            )

        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        actual_smiles = canonicalize_smiles(validated_data.smiles, ignore_stereo=ignore_stereo)
        if actual_smiles != expected_smiles:
            raise adapter_target_mismatch(
                "retrochimera",
                target.id,
                expected_smiles=expected_smiles,
                actual_smiles=actual_smiles,
            )

        if validated_data.result.outputs is None:
            raise adapter_route_transform_error(
                "retrochimera",
                target.id,
                "validated payload is missing result outputs",
                payload_field="result.outputs",
            )

        rank = 1
        for output in validated_data.result.outputs:
            for route in output.routes:
                try:
                    route_obj = self._transform(route, target, rank=rank, ignore_stereo=ignore_stereo)
                    yield route_obj
                    rank += 1
                except RetroCastException as e:
                    logger.warning(f"  - route for '{target.id}' failed transformation: {e} [{e.code}]")
                    continue

    def _transform(
        self, route: RetrochimeraRoute, target: TargetIdentity, rank: int, ignore_stereo: bool = False
    ) -> Route:
        """
        orchestrates the transformation of a single retrochimera route.
        raises RetroCastException on failure.
        """
        precursor_map = self._build_precursor_map(route, ignore_stereo=ignore_stereo)
        target_molecule = build_molecule_from_precursor_map(
            smiles=SmilesStr(target.smiles),
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter="retrochimera",
        )

        return Route(target=target_molecule, rank=rank, metadata={})

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
