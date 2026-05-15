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
class DreamRetroRoutePayload:
    route_str: str
    metadata: dict[str, Any]


class DreamRetroAdapter(BaseAdapter):
    """adapter for converting dreamretro-style outputs to the route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
        expected_target: TargetIdentity | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw DreamRetro data and expose the single route-like payload.
        """
        target_id = expected_target.id if expected_target is not None else source_key or "<unknown>"
        if not isinstance(raw_data, dict):
            raise adapter_schema_error("dreamretro", target_id, "expected a dict")

        if not raw_data.get("succ"):
            logger.debug(f"skipping raw data for '{target_id}': 'succ' is not true.")
            return

        route_str = raw_data.get("routes")
        if not isinstance(route_str, str) or not route_str:
            raise adapter_schema_error("dreamretro", target_id, "no valid 'routes' string found")

        metadata = {
            key: raw_data[key]
            for key in ["expand_model_call", "value_model_call", "reaction_nodes_lens", "mol_nodes_lens"]
            if key in raw_data
        }
        yield RawRouteEntry(
            payload=DreamRetroRoutePayload(route_str=route_str, metadata=metadata),
            source_key=source_key,
            expected_target_id=expected_target.id if expected_target is not None else None,
            expected_target_smiles=expected_target.smiles if expected_target is not None else None,
            source_order=1,
        )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, DreamRetroRoutePayload):
            raise adapter_schema_error(
                "dreamretro",
                expected_target.id if expected_target is not None else "<unknown>",
                "expected a dreamretro route payload",
            )
        return self._transform(
            raw_route.route_str,
            expected_target,
            raw_route.metadata,
            ignore_stereo=ignore_stereo,
        )

    def _parse_route_string(self, route_str: str, ignore_stereo: bool = False) -> tuple[SmilesStr, PrecursorMap]:
        """
        parses the dreamretro route string into a target smiles and a precursor map.

        raises:
            adapterlogicerror: if the string format is invalid.
        """
        precursor_map: PrecursorMap = {}
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise adapter_route_string_error("dreamretro", "empty route string", empty=True)

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
                "dreamretro",
                "expected each reaction step to split into product, reagents, and reactants",
                fragment=current_step_for_error_reporting[:70],
            ) from e

    def _transform(
        self,
        route_str: str,
        target_input: TargetIdentity | None,
        metadata: dict[str, Any],
        ignore_stereo: bool = False,
    ) -> Route:
        """
        orchestrates the transformation of a single dreamretro route string.
        """
        parsed_target_smiles, precursor_map = self._parse_route_string(route_str, ignore_stereo=ignore_stereo)

        if target_input is not None:
            expected_smiles = canonicalize_smiles(target_input.smiles, ignore_stereo=ignore_stereo)
            if parsed_target_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "dreamretro",
                    target_input.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=parsed_target_smiles,
                )
            target_smiles = expected_smiles
        else:
            target_smiles = parsed_target_smiles

        molecule = build_molecule_from_precursor_map(
            smiles=target_smiles,
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter="dreamretro",
        )

        return Route(target=molecule, metadata=metadata)
