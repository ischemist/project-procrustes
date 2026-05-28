from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from numbers import Real
from typing import Any

from retrocast.adapters.errors import (
    adapter_route_string_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.typing import SmilesStr
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.adapters.common import build_molecule_from_precursor_map
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target

# SECTION: Raw RetroStar Schema


@dataclass(frozen=True, slots=True)
class RetroStarRoutePayload:
    route_str: str
    route_cost: float | None


@dataclass(slots=True)
class RetroStarParsedRoute:
    target_smiles: SmilesStr
    precursor_map: dict[SmilesStr, list[str]]
    step_scores: dict[SmilesStr, float]


# SECTION: Adapter


class RetroStarAdapter:
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        if not isinstance(raw_payload, dict):
            raise adapter_schema_error("retrostar", target_id, "expected a dict")

        if not raw_payload.get("succ"):
            return

        route_str = raw_payload.get("routes")
        if not isinstance(route_str, str) or not route_str:
            raise adapter_schema_error("retrostar", target_id, "no valid 'routes' string found")

        route_cost = raw_payload.get("route_cost")
        if isinstance(route_cost, bool) or not isinstance(route_cost, Real):
            route_cost = None
        yield RawRouteEntry(
            payload=RetroStarRoutePayload(
                route_str=route_str,
                route_cost=float(route_cost) if route_cost is not None else None,
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
        if not isinstance(raw_route, RetroStarRoutePayload):
            raise adapter_schema_error("retrostar", target_id, "expected a retrostar route payload")

        parsed_route = self._parse_route_string(raw_route.route_str, mode=mode)

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if parsed_route.target_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "retrostar",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=parsed_route.target_smiles,
                )

        route_target = build_molecule_from_precursor_map(
            parsed_route.target_smiles,
            parsed_route.precursor_map,
            adapter="retrostar",
            mode=mode,
            reaction_annotations={
                product_smiles: {"step_score": step_score}
                for product_smiles, step_score in parsed_route.step_scores.items()
            },
        )
        if route_target is None:
            raise AdapterLogicError(
                "RetroStar target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "retrostar", "target_id": target_id},
            )

        annotations: dict[str, Any] = {}
        if raw_route.route_cost is not None:
            annotations["route_cost"] = raw_route.route_cost
        return Route(target=route_target, annotations=annotations)

    def _parse_route_string(self, route_str: str, *, mode: AdaptMode = "strict") -> RetroStarParsedRoute:
        steps = route_str.split("|")
        if not steps or not steps[0]:
            raise adapter_route_string_error("retrostar", "empty route string", empty=True)

        if len(steps) == 1 and ">" not in steps[0]:
            return RetroStarParsedRoute(target_smiles=canonicalize_smiles(steps[0]), precursor_map={}, step_scores={})

        precursor_map: dict[SmilesStr, list[str]] = {}
        step_scores: dict[SmilesStr, float] = {}
        current_step = ""
        try:
            first_product_smiles = steps[0].split(">")[0]
            target_smiles = canonicalize_smiles(first_product_smiles)

            for step in steps:
                current_step = step
                parts = step.split(">")
                if len(parts) != 3:
                    raise ValueError("invalid step format")
                product_smiles, score_text, reactants_smiles = parts
                try:
                    canon_product = canonicalize_smiles(product_smiles)
                except InvalidSmilesError:
                    if mode == "prune" and product_smiles != first_product_smiles:
                        continue
                    raise
                reactants = [reactant.strip() for reactant in reactants_smiles.split(".") if reactant.strip()]
                precursor_map[canon_product] = reactants
                with suppress(ValueError):
                    step_scores[canon_product] = float(score_text)

            return RetroStarParsedRoute(
                target_smiles=target_smiles,
                precursor_map=precursor_map,
                step_scores=step_scores,
            )
        except (ValueError, IndexError) as exc:
            if isinstance(exc, InvalidSmilesError):
                raise
            raise adapter_route_string_error(
                "retrostar",
                "expected each reaction step to split into product, reagents, and reactants",
                fragment=current_step[:70],
            ) from exc
