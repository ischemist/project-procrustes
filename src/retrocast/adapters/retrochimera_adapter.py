from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_tree_from_precursor_map
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import BenchmarkTree, TargetInfo
from retrocast.exceptions import UrsaException
from retrocast.utils.logging import logger

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
    """adapter for converting retrochimera-style outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates raw retrochimera data, transforms it, and yields benchmarktree objects.
        """
        try:
            validated_data = RetrochimeraData.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(
                f"  - raw data for target '{target_info.id}' failed retrochimera schema validation. error: {e}"
            )
            return

        if validated_data.result.error is not None:
            error_msg = validated_data.result.error.get("message", "unknown error")
            error_type = validated_data.result.error.get("type", "unknown")
            logger.warning(
                f"  - retrochimera reported an error for target '{target_info.id}': {error_type} - {error_msg}"
            )
            return

        if canonicalize_smiles(validated_data.smiles) != target_info.smiles:
            logger.warning(
                f"  - mismatched smiles for target '{target_info.id}': expected {target_info.smiles}, got {canonicalize_smiles(validated_data.smiles)}"
            )
            return

        if validated_data.result.outputs is None:
            logger.warning(f"  - no outputs found for target '{target_info.id}'")
            return

        for output in validated_data.result.outputs:
            for route in output.routes:
                try:
                    tree = self._transform(route, target_info)
                    yield tree
                except UrsaException as e:
                    logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                    continue

    def _transform(self, route: RetrochimeraRoute, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single retrochimera route.
        raises ursaexception on failure.
        """
        precursor_map = self._build_precursor_map(route)
        # refactor: use the common recursive builder.
        retrosynthetic_tree = build_tree_from_precursor_map(smiles=target_info.smiles, precursor_map=precursor_map)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _build_precursor_map(self, route: RetrochimeraRoute) -> PrecursorMap:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: PrecursorMap = {}
        for reaction in route.reactions:
            canon_product = canonicalize_smiles(reaction.product)
            canon_reactants = [canonicalize_smiles(r) for r in reaction.reactants]
            precursor_map[canon_product] = canon_reactants
        return precursor_map
