from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_string_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.models.route import Route
from retrocast.models.task import Target
from retrocast.typing import SmilesStr

# SECTION: Raw DreamRetro Schema


@dataclass(frozen=True, slots=True)
class DreamRetroRoutePayload:
    route_str: str
    annotations: tuple[tuple[str, Any], ...]

    def route_annotations(self) -> dict[str, Any]:
        return dict(self.annotations)


@dataclass(slots=True)
class DreamRetroParsedRoute:
    target_smiles: SmilesStr
    precursor_map: dict[SmilesStr, list[str]]


# SECTION: Adapter


class DreamRetroErAdapter:
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        if not isinstance(raw_payload, dict):
            raise adapter_schema_error("dreamretro", target_id, "expected a dict")
        if not raw_payload.get("succ"):
            return

        route_str = raw_payload.get("routes")
        if not isinstance(route_str, str) or not route_str:
            raise adapter_schema_error("dreamretro", target_id, "no valid 'routes' string found")

        annotations = {
            key: raw_payload[key]
            for key in ("expand_model_call", "value_model_call", "reaction_nodes_lens", "mol_nodes_lens")
            if key in raw_payload
        }
        yield RawRouteEntry(
            payload=DreamRetroRoutePayload(
                route_str=route_str,
                annotations=tuple((key, deepcopy(value)) for key, value in annotations.items()),
            ),
            source_key=source_key,
            source_order=1,
        )

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        if not isinstance(raw_route, DreamRetroRoutePayload):
            raise adapter_schema_error("dreamretro", target_id, "expected a dreamretro route payload")

        parsed_route = self._parse_route_string(raw_route.route_str, mode=mode)
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if parsed_route.target_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "dreamretro",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=parsed_route.target_smiles,
                )

        route_target = build_molecule_from_precursor_map(
            parsed_route.target_smiles,
            parsed_route.precursor_map,
            adapter="dreamretro",
            mode=mode,
        )
        if route_target is None:
            raise AdapterLogicError(
                "DreamRetro target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "dreamretro", "target_id": target_id},
            )
        return Route(target=route_target, annotations=raw_route.route_annotations())

    def _parse_route_string(self, route_str: str, *, mode: AdaptMode = "strict") -> DreamRetroParsedRoute:
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise adapter_route_string_error("dreamretro", "empty route string", empty=True)

        if len(steps) == 1 and ">" not in steps[0]:
            return DreamRetroParsedRoute(target_smiles=canonicalize_smiles(steps[0]), precursor_map={})

        precursor_map: dict[SmilesStr, list[str]] = {}
        current_step = ""
        try:
            first_product_smiles = steps[0].split(">")[0]
            target_smiles = canonicalize_smiles(first_product_smiles)
            for step in steps:
                current_step = step
                parts = step.split(">")
                if len(parts) != 3:
                    raise ValueError("invalid step format")
                product_smiles, _, reactants_smiles = parts
                try:
                    canon_product = canonicalize_smiles(product_smiles)
                except InvalidSmilesError:
                    if mode == "prune" and product_smiles != first_product_smiles:
                        continue
                    raise
                precursor_map[canon_product] = [
                    reactant.strip() for reactant in reactants_smiles.split(".") if reactant.strip()
                ]
            return DreamRetroParsedRoute(target_smiles=target_smiles, precursor_map=precursor_map)
        except (ValueError, IndexError) as exc:
            if isinstance(exc, InvalidSmilesError):
                raise
            raise adapter_route_string_error(
                "dreamretro",
                "expected each reaction step to split into product, reagents, and reactants",
                fragment=current_step[:70],
            ) from exc
