from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_string_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetroStarRoutePayload:
    route_str: str
    route_cost: float | None


class RetroStarAdapter(BaseAdapter):
    """Adapter for converting RetroStar-style outputs to the Route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw RetroStar data and expose the single route-like payload.
        """
        target_id = source_key or "<unknown>"
        if not isinstance(raw_data, dict):
            raise adapter_schema_error("retrostar", target_id, "expected a dict")

        if not raw_data.get("succ"):
            logger.debug(f"Skipping raw data for '{target_id}': 'succ' is not true.")
            return

        route_str = raw_data.get("routes")
        if not isinstance(route_str, str) or not route_str:
            raise adapter_schema_error("retrostar", target_id, "no valid 'routes' string found")

        yield RawRouteEntry(
            payload=RetroStarRoutePayload(route_str=route_str, route_cost=raw_data.get("route_cost")),
            source_key=source_key,
            target_hint_id=None,
            target_hint_smiles=None,
            source_order=1,
        )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, RetroStarRoutePayload):
            raise adapter_schema_error(
                "retrostar",
                expected_target.id if expected_target is not None else "<unknown>",
                "expected a retrostar route payload",
            )
        return self._transform(
            raw_route.route_str,
            expected_target,
            route_cost=raw_route.route_cost,
            ignore_stereo=ignore_stereo,
        )

    def _parse_route_string(self, route_str: str, ignore_stereo: bool = False) -> tuple[SmilesStr, PrecursorMap]:
        """
        Parses the RetroStar route string into a target SMILES and a precursor map.

        Raises:
            AdapterLogicError: If the string format is invalid.
        """
        precursor_map: PrecursorMap = {}
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise adapter_route_string_error("retrostar", "empty route string", empty=True)

        if len(steps) == 1 and ">" not in steps[0]:
            target_smiles = canonicalize_smiles(steps[0], ignore_stereo=ignore_stereo)
            return target_smiles, {}

        current_step_for_error_reporting = ""
        try:
            current_step_for_error_reporting = steps[0]
            if len(current_step_for_error_reporting.split(">")) != 3:
                raise ValueError("invalid step format")
            first_product_smiles, _, _ = current_step_for_error_reporting.split(">")
            target_smiles = canonicalize_smiles(first_product_smiles, ignore_stereo=ignore_stereo)

            for step in steps:
                current_step_for_error_reporting = step
                parts = step.split(">")
                if len(parts) != 3:
                    raise ValueError("invalid step format")
                product_smi, _, reactants_smi = parts

                full_canonical_reactants = canonicalize_smiles(reactants_smi, ignore_stereo=ignore_stereo)
                canon_product = canonicalize_smiles(product_smi, ignore_stereo=ignore_stereo)
                precursor_map[canon_product] = [SmilesStr(s) for s in str(full_canonical_reactants).split(".")]

            return target_smiles, precursor_map
        except (ValueError, IndexError) as e:
            raise adapter_route_string_error(
                "retrostar",
                "expected each reaction step to split into product, reagents, and reactants",
                fragment=current_step_for_error_reporting[:70],
            ) from e

    def _transform(
        self,
        route_str: str,
        target: TargetIdentity | None,
        route_cost: float | None = None,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        Orchestrates the transformation of a single RetroStar route string.
        Raises RetroCastException on failure.
        """
        parsed_target_smiles, precursor_map = self._parse_route_string(route_str, ignore_stereo=ignore_stereo)

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if parsed_target_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "retrostar",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=parsed_target_smiles,
                )
            target_smiles = SmilesStr(target.smiles)
        else:
            target_smiles = parsed_target_smiles

        # Build the molecule tree using the new schema helper
        target_molecule = build_molecule_from_precursor_map(
            smiles=target_smiles,
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter="retrostar",
        )

        # Build metadata
        metadata = {}
        if route_cost is not None:
            metadata["route_cost"] = route_cost

        return Route(target=target_molecule, metadata=metadata)
