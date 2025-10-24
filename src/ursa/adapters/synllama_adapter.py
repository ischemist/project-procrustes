""" """

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from ursa.adapters.base_adapter import BaseAdapter
from ursa.adapters.common import PrecursorMap, build_tree_from_precursor_map
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import BenchmarkTree, TargetInfo
from ursa.exceptions import AdapterLogicError, UrsaException
from ursa.utils.logging import logger

# --- pydantic models for input validation ---


class SynLlamaRouteInput(BaseModel):
    synthesis_string: str


class SynLlamaRouteList(RootModel[list[SynLlamaRouteInput]]):
    pass


class SynLlaMaAdapter(BaseAdapter):
    """adapter for converting pre-processed synllama outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """validates the pre-processed json data for synllama and yields benchmarktree objects."""
        try:
            validated_routes = SynLlamaRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - data for target '{target_info.id}' failed synllama schema validation. error: {e}")
            return

        for route in validated_routes.root:
            try:
                tree = self._transform(route, target_info)
                yield tree
            except UrsaException as e:
                logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, route: SynLlamaRouteInput, target_info: TargetInfo) -> BenchmarkTree:
        """orchestrates the transformation of a single synllama route string."""
        precursor_map = self._parse_synthesis_string(route.synthesis_string)

        # the product from the string should match our target
        parsed_target_smiles = next(iter(precursor_map.keys()))

        if parsed_target_smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        retrosynthetic_tree = build_tree_from_precursor_map(smiles=target_info.smiles, precursor_map=precursor_map)
        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _parse_synthesis_string(self, synthesis_str: str) -> PrecursorMap:
        """
        parses the synllama route string into a precursor map.
        format is `reactant1;reactant2;...;template_id;product`.
        """
        parts = synthesis_str.split(";")
        if len(parts) < 3:
            raise AdapterLogicError(f"invalid synthesis string format: '{synthesis_str}'")

        product = canonicalize_smiles(parts[-1])
        # the reaction template is at [-2], reactants are everything before that.
        reactants = [canonicalize_smiles(r) for r in parts[:-2]]

        return {product: reactants}
