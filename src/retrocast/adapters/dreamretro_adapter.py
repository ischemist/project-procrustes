from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


class DreamRetroAdapter(BaseAdapter):
    """adapter for converting dreamretro-style outputs to the route schema."""

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        validates raw dreamretro data, transforms its single route string, and yields a route.
        """
        if not isinstance(raw_target_data, dict):
            logger.warning(
                f"  - raw data for target '{target.id}' failed validation: expected a dict, got {type(raw_target_data).__name__}."
            )
            return

        if not raw_target_data.get("succ"):
            logger.debug(f"skipping raw data for '{target.id}': 'succ' is not true.")
            return

        route_str = raw_target_data.get("routes")
        if not isinstance(route_str, str) or not route_str:
            logger.warning(f"  - raw data for target '{target.id}' failed validation: no valid 'routes' string found.")
            return

        try:
            route = self._transform(route_str, target, raw_target_data, ignore_stereo=ignore_stereo)
            yield route
        except RetroCastException as e:
            logger.warning(f"  - route for '{target.id}' failed transformation: {e}")
            return

    def _parse_route_string(self, route_str: str, ignore_stereo: bool = False) -> tuple[SmilesStr, PrecursorMap]:
        """
        parses the dreamretro route string into a target smiles and a precursor map.

        raises:
            adapterlogicerror: if the string format is invalid.
        """
        precursor_map: PrecursorMap = {}
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise AdapterLogicError("route string is empty or invalid.")

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
            raise AdapterLogicError(
                f"failed to parse route string step. invalid format near '{current_step_for_error_reporting[:70]}...'."
            ) from e

    def _transform(
        self, route_str: str, target_input: TargetIdentity, raw_data: dict[str, Any], ignore_stereo: bool = False
    ) -> Route:
        """
        orchestrates the transformation of a single dreamretro route string.
        """
        parsed_target_smiles, precursor_map = self._parse_route_string(route_str, ignore_stereo=ignore_stereo)

        if parsed_target_smiles != target_input.smiles:
            msg = (
                f"mismatched smiles for target {target_input.id}. "
                f"expected canonical: {target_input.smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        # build molecule tree from precursor map
        molecule = build_molecule_from_precursor_map(
            smiles=SmilesStr(target_input.smiles), precursor_map=precursor_map, ignore_stereo=ignore_stereo
        )

        # extract metadata
        metadata = {
            key: raw_data[key]
            for key in ["expand_model_call", "value_model_call", "reaction_nodes_lens", "mol_nodes_lens"]
            if key in raw_data
        }

        return Route(target=molecule, rank=1, metadata=metadata)
