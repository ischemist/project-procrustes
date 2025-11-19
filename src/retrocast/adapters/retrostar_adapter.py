from __future__ import annotations

from collections.abc import Generator
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr
from retrocast.utils.logging import logger


class RetroStarAdapter(BaseAdapter):
    """Adapter for converting RetroStar-style outputs to the Route schema."""

    def cast(self, raw_target_data: Any, target_input: TargetIdentity) -> Generator[Route, None, None]:
        """
        Validates raw RetroStar data, transforms its single route string, and yields a Route.
        """
        if not isinstance(raw_target_data, dict):
            logger.warning(
                f"  - Raw data for target '{target_input.id}' failed validation: expected a dict, got {type(raw_target_data).__name__}."
            )
            return

        if not raw_target_data.get("succ"):
            logger.debug(f"Skipping raw data for '{target_input.id}': 'succ' is not true.")
            return

        route_str = raw_target_data.get("routes")
        if not isinstance(route_str, str) or not route_str:
            logger.warning(
                f"  - Raw data for target '{target_input.id}' failed validation: no valid 'routes' string found."
            )
            return

        # Extract route_cost if available
        route_cost = raw_target_data.get("route_cost")

        try:
            route = self._transform(route_str, target_input, route_cost=route_cost)
            yield route
        except RetroCastException as e:
            logger.warning(f"  - Route for '{target_input.id}' failed transformation: {e}")
            return

    def _parse_route_string(self, route_str: str) -> tuple[SmilesStr, PrecursorMap]:
        """
        Parses the RetroStar route string into a target SMILES and a precursor map.

        Raises:
            AdapterLogicError: If the string format is invalid.
        """
        precursor_map: PrecursorMap = {}
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise AdapterLogicError("Route string is empty or invalid.")

        if len(steps) == 1 and ">" not in steps[0]:
            target_smiles = canonicalize_smiles(steps[0])
            return target_smiles, {}

        current_step_for_error_reporting = ""
        try:
            current_step_for_error_reporting = steps[0]
            if len(current_step_for_error_reporting.split(">")) != 3:
                raise ValueError("Invalid step format")
            first_product_smiles, _, _ = current_step_for_error_reporting.split(">")
            target_smiles = canonicalize_smiles(first_product_smiles)

            for step in steps:
                current_step_for_error_reporting = step
                parts = step.split(">")
                if len(parts) != 3:
                    raise ValueError("Invalid step format")
                product_smi, _, reactants_smi = parts

                full_canonical_reactants = canonicalize_smiles(reactants_smi)
                canon_product = canonicalize_smiles(product_smi)
                precursor_map[canon_product] = [SmilesStr(s) for s in str(full_canonical_reactants).split(".")]

            return target_smiles, precursor_map
        except (ValueError, IndexError) as e:
            raise AdapterLogicError(
                f"Failed to parse route string step. Invalid format near '{current_step_for_error_reporting[:70]}...'."
            ) from e

    def _transform(self, route_str: str, target_input: TargetIdentity, route_cost: float | None = None) -> Route:
        """
        Orchestrates the transformation of a single RetroStar route string.
        Raises RetroCastException on failure.
        """
        parsed_target_smiles, precursor_map = self._parse_route_string(route_str)

        if parsed_target_smiles != target_input.smiles:
            msg = (
                f"Mismatched SMILES for target {target_input.id}. "
                f"Expected canonical: {target_input.smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        # Build the molecule tree using the new schema helper
        target_molecule = build_molecule_from_precursor_map(
            smiles=SmilesStr(target_input.smiles), precursor_map=precursor_map
        )

        # Build metadata
        metadata = {}
        if route_cost is not None:
            metadata["route_cost"] = route_cost

        # RetroStar produces a single route per target, so rank is always 1
        return Route(target=target_molecule, rank=1, metadata=metadata)
