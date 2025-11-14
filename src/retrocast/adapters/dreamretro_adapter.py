from __future__ import annotations

from collections.abc import Generator
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_tree_from_precursor_map
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import BenchmarkTree, TargetInfo
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.typing import SmilesStr
from retrocast.utils.logging import logger


class DreamRetroAdapter(BaseAdapter):
    """adapter for converting dreamretro-style outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates raw dreamretro data, transforms its single route string, and yields a benchmarktree.
        """
        if not isinstance(raw_target_data, dict):
            logger.warning(
                f"  - raw data for target '{target_info.id}' failed validation: expected a dict, got {type(raw_target_data).__name__}."
            )
            return

        if not raw_target_data.get("succ"):
            logger.debug(f"skipping raw data for '{target_info.id}': 'succ' is not true.")
            return

        route_str = raw_target_data.get("routes")
        if not isinstance(route_str, str) or not route_str:
            logger.warning(
                f"  - raw data for target '{target_info.id}' failed validation: no valid 'routes' string found."
            )
            return

        try:
            tree = self._transform(route_str, target_info)
            yield tree
        except RetroCastException as e:
            logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
            return

    def _parse_route_string(self, route_str: str) -> tuple[SmilesStr, PrecursorMap]:
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
            target_smiles = canonicalize_smiles(steps[0])
            return target_smiles, {}

        current_step_for_error_reporting = ""
        try:
            current_step_for_error_reporting = steps[0]
            if len(current_step_for_error_reporting.split(">")) != 3:
                raise ValueError("invalid step format")
            first_product_smiles, _, _ = current_step_for_error_reporting.split(">")
            target_smiles = canonicalize_smiles(first_product_smiles)

            for step in steps:
                current_step_for_error_reporting = step
                parts = step.split(">")
                if len(parts) != 3:
                    raise ValueError("invalid step format")
                product_smi, _, reactants_smi = parts

                full_canonical_reactants = canonicalize_smiles(reactants_smi)
                canon_product = canonicalize_smiles(product_smi)
                precursor_map[canon_product] = [SmilesStr(s) for s in str(full_canonical_reactants).split(".")]

            return target_smiles, precursor_map
        except (ValueError, IndexError) as e:
            raise AdapterLogicError(
                f"failed to parse route string step. invalid format near '{current_step_for_error_reporting[:70]}...'."
            ) from e

    def _transform(self, route_str: str, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single dreamretro route string.
        """
        parsed_target_smiles, precursor_map = self._parse_route_string(route_str)

        if parsed_target_smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        # refactor: use the common recursive builder.
        retrosynthetic_tree = build_tree_from_precursor_map(smiles=target_info.smiles, precursor_map=precursor_map)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)
