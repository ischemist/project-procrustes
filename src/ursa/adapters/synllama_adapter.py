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
        # the final product is always the last element in the semicolon-delimited string.
        # this is the most reliable way to identify it.
        synthesis_parts = [p.strip() for p in route.synthesis_string.split(";") if p.strip()]
        if not synthesis_parts:
            raise AdapterLogicError("synthesis string is empty.")

        # the final product is always the last element. this is the most reliable way to identify it.
        parsed_target_smiles = canonicalize_smiles(synthesis_parts[-1])
        if parsed_target_smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        precursor_map = self._parse_synthesis_string(route.synthesis_string)
        retrosynthetic_tree = build_tree_from_precursor_map(smiles=target_info.smiles, precursor_map=precursor_map)
        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _parse_synthesis_string(self, synthesis_str: str) -> PrecursorMap:
        """
        parses a multi-step synllama route string into a precursor map.
        the format is a sequence of `reactants;template;product` chunks, chained together.
        e.g., r1;r2;t1;p1;r3;t2;p2 means p1=f(r1,r2) and p2=f(p1,r3).
        """
        precursor_map: PrecursorMap = {}
        # clean up parts: remove whitespace and empty strings from sequences like ';;'
        parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]

        if not parts:
            raise AdapterLogicError("synthesis string is empty.")

        template_indices = [i for i, p in enumerate(parts) if p.startswith("R") and p[1:].isdigit()]

        if not template_indices:
            # if no templates, assume no reactions. it's a purchasable molecule.
            return precursor_map

        last_product_smi = None
        reactant_start_idx = 0
        for template_idx in template_indices:
            product_idx = template_idx + 1
            if product_idx >= len(parts):
                raise AdapterLogicError(f"malformed route: template '{parts[template_idx]}' has no product.")

            product_smiles = canonicalize_smiles(parts[product_idx])
            explicit_reactant_parts = parts[reactant_start_idx:template_idx]
            all_reactants = [canonicalize_smiles(r) for r in explicit_reactant_parts]
            if last_product_smi:
                all_reactants.append(last_product_smi)

            if not all_reactants:
                raise AdapterLogicError(f"no reactants found for product '{parts[product_idx]}'")

            precursor_map[product_smiles] = all_reactants

            last_product_smi = product_smiles
            reactant_start_idx = product_idx + 1

        return precursor_map
