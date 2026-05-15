from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_string_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---


class SynLlamaRouteInput(BaseModel):
    synthesis_string: str


class SynLlamaRouteList(RootModel[list[SynLlamaRouteInput]]):
    pass


class SynLlaMaAdapter(BaseAdapter):
    """adapter for converting pre-processed synllama outputs to the Route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
        expected_target: TargetIdentity | None = None,
    ) -> Iterator[RawRouteEntry]:
        """Validate raw SynLlama data and expose one route-like payload per entry."""
        target_id = expected_target.id if expected_target is not None else source_key or "<unknown>"
        try:
            validated_routes = SynLlamaRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("synllama", target_id, "invalid route list") from e

        for rank, route in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=route,
                source_key=source_key,
                expected_target_id=expected_target.id if expected_target is not None else None,
                expected_target_smiles=expected_target.smiles if expected_target is not None else None,
                source_order=rank,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, SynLlamaRouteInput):
            raw_route = SynLlamaRouteInput.model_validate(raw_route)
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(
        self,
        route: SynLlamaRouteInput,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
    ) -> Route:
        """orchestrates the transformation of a single synllama route string."""
        # the final product is always the last element in the semicolon-delimited string.
        # this is the most reliable way to identify it.
        synthesis_parts = [p.strip() for p in route.synthesis_string.split(";") if p.strip()]
        if not synthesis_parts:
            raise adapter_route_string_error("synllama", "empty synthesis string", empty=True)

        # the final product is always the last element. this is the most reliable way to identify it.
        parsed_target_smiles = canonicalize_smiles(synthesis_parts[-1], ignore_stereo=ignore_stereo)
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if parsed_target_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "synllama",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=parsed_target_smiles,
                )
            target_smiles = SmilesStr(target.smiles)
        else:
            target_smiles = parsed_target_smiles

        precursor_map = self._parse_synthesis_string(route.synthesis_string, ignore_stereo=ignore_stereo)
        target_molecule = build_molecule_from_precursor_map(
            smiles=target_smiles,
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter="synllama",
        )
        return Route(target=target_molecule, metadata={})

    def _parse_synthesis_string(
        self, synthesis_str: str, ignore_stereo: bool = False
    ) -> dict[SmilesStr, list[SmilesStr]]:
        """
        parses a multi-step synllama route string into a precursor map.
        the format is a sequence of `reactants;template;product` chunks, chained together.
        e.g., r1;r2;t1;p1;r3;t2;p2 means p1=f(r1,r2) and p2=f(p1,r3).
        """
        precursor_map: dict[SmilesStr, list[SmilesStr]] = {}
        # clean up parts: remove whitespace and empty strings from sequences like ';;'
        parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]

        if not parts:
            raise adapter_route_string_error("synllama", "empty synthesis string", empty=True)

        template_indices = [i for i, p in enumerate(parts) if p.startswith("R") and p[1:].isdigit()]

        if not template_indices:
            # if no templates, assume no reactions. it's a purchasable molecule.
            return precursor_map

        last_product_smi = None
        reactant_start_idx = 0
        for template_idx in template_indices:
            product_idx = template_idx + 1
            if product_idx >= len(parts):
                raise adapter_route_string_error(
                    "synllama",
                    "template has no product",
                    fragment=parts[template_idx],
                )

            product_smiles = canonicalize_smiles(parts[product_idx], ignore_stereo=ignore_stereo)
            explicit_reactant_parts = parts[reactant_start_idx:template_idx]
            all_reactants = [canonicalize_smiles(r, ignore_stereo=ignore_stereo) for r in explicit_reactant_parts]
            if last_product_smi:
                all_reactants.append(last_product_smi)

            if not all_reactants:
                raise adapter_route_string_error(
                    "synllama",
                    "no reactants found for product",
                    fragment=parts[product_idx],
                )

            precursor_map[product_smiles] = all_reactants

            last_product_smi = product_smiles
            reactant_start_idx = product_idx + 1

        return precursor_map
