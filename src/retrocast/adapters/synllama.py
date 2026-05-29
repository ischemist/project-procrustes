from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_string_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError
from retrocast.models.route import Route
from retrocast.models.task import Target
from retrocast.typing import SmilesStr

# SECTION: Raw SynLlama Schema


class SynLlamaRouteInput(BaseModel):
    synthesis_string: str


class SynLlamaRouteList(RootModel[list[SynLlamaRouteInput]]):
    pass


# SECTION: Adapter


class SynLlamaAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            routes = SynLlamaRouteList.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("synllama", target_id, "invalid route list") from exc
        for source_order, route in enumerate(routes.root, start=1):
            yield RawRouteEntry(payload=route, source_key=source_key, source_order=source_order)

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        try:
            route = SynLlamaRouteInput.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("synllama", target_id, "invalid route") from exc
        parts = [part.strip() for part in route.synthesis_string.split(";") if part.strip()]
        if not parts:
            raise adapter_route_string_error("synllama", "empty synthesis string", empty=True)
        try:
            parsed_target = canonicalize_smiles(parts[-1])
        except InvalidSmilesError:
            if mode == "prune":
                raise AdapterLogicError(
                    "SynLlama target molecule was pruned",
                    code="adapter.target_pruned",
                    context={"adapter": "synllama", "target_id": target_id},
                ) from None
            raise
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if parsed_target != expected_smiles:
                raise adapter_target_mismatch(
                    "synllama", target.id, expected_smiles=expected_smiles, actual_smiles=parsed_target
                )
        precursor_map = self._parse_synthesis_string(route.synthesis_string, mode=mode)
        route_target = build_molecule_from_precursor_map(parsed_target, precursor_map, adapter="synllama", mode=mode)
        if route_target is None:
            raise AdapterLogicError(
                "SynLlama target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "synllama", "target_id": target_id},
            )
        return Route(target=route_target)

    def _parse_synthesis_string(self, synthesis_str: str, *, mode: AdaptMode = "strict") -> dict[SmilesStr, list[str]]:
        parts = [part.strip() for part in synthesis_str.split(";") if part.strip()]
        if not parts:
            raise adapter_route_string_error("synllama", "empty synthesis string", empty=True)

        template_indices = [index for index, part in enumerate(parts) if part.startswith("R") and part[1:].isdigit()]
        if not template_indices:
            return {}

        precursor_map: dict[SmilesStr, list[str]] = {}
        last_product_smiles: SmilesStr | None = None
        reactant_start = 0
        for template_index in template_indices:
            product_index = template_index + 1
            if product_index >= len(parts):
                raise adapter_route_string_error("synllama", "template has no product", fragment=parts[template_index])
            try:
                product_smiles = canonicalize_smiles(parts[product_index])
            except InvalidSmilesError:
                if mode == "strict":
                    raise
                last_product_smiles = None
                reactant_start = product_index + 1
                continue
            reactants = parts[reactant_start:template_index]
            if last_product_smiles is not None:
                reactants.append(last_product_smiles)
            if not reactants:
                raise adapter_route_string_error(
                    "synllama", "no reactants found for product", fragment=parts[product_index]
                )
            precursor_map[product_smiles] = reactants
            last_product_smiles = product_smiles
            reactant_start = product_index + 1
        return precursor_map
