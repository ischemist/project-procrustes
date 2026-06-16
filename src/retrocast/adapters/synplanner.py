from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.common import build_bipartite_molecule
from retrocast.adapters.errors import adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError
from retrocast.models.route import Route
from retrocast.models.task import Target

logger = logging.getLogger(__name__)

# SECTION: Raw SynPlanner Schema


class SynPlannerBaseNode(BaseModel):
    smiles: str
    children: list[SynPlannerNode] = Field(default_factory=list)


class SynPlannerMoleculeInput(SynPlannerBaseNode):
    type: Literal["mol"]
    in_stock: bool = False


class SynPlannerReactionInput(SynPlannerBaseNode):
    type: Literal["reaction"]


SynPlannerNode = Annotated[SynPlannerMoleculeInput | SynPlannerReactionInput, Field(discriminator="type")]


# SECTION: Adapter


class SynPlannerAdapter:
    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        if not isinstance(raw_payload, Sequence) or isinstance(raw_payload, (str, bytes, bytearray)):
            raise adapter_schema_error(
                "synplanner",
                target_id,
                "invalid route list",
                raw_type=type(raw_payload).__name__,
            )

        valid_count = 0
        invalid_count = 0
        first_invalid: tuple[int, int, str] | None = None
        for source_order, route_payload in enumerate(raw_payload, start=1):
            try:
                route_root = SynPlannerMoleculeInput.model_validate(route_payload)
            except ValidationError as exc:
                source_index = source_order - 1
                errors = exc.errors(include_url=False)
                if errors:
                    first_error = errors[0]
                    raw_location = first_error.get("loc", ())
                    location = (
                        ".".join(str(part) for part in raw_location)
                        if isinstance(raw_location, tuple | list)
                        else str(raw_location)
                    )
                    message = str(first_error.get("msg", "validation failed"))
                    summary = f"{location}: {message}" if location else message
                else:
                    summary = "validation failed"
                if len(errors) > 1:
                    summary = f"{summary} (+ {len(errors) - 1} more)"
                invalid_count += 1
                if first_invalid is None:
                    first_invalid = (source_index, source_order, summary)
                logger.warning(
                    "skipping invalid synplanner route: target=%s source_index=%s source_order=%s error=%s",
                    target_id,
                    source_index,
                    source_order,
                    summary,
                )
                continue

            valid_count += 1
            yield RawRouteEntry(payload=route_root, source_key=source_key, source_order=source_order)

        if invalid_count:
            logger.warning(
                "skipped %s invalid synplanner route(s) for target %s; valid_routes=%s",
                invalid_count,
                target_id,
                valid_count,
            )
        if valid_count == 0 and first_invalid is not None:
            first_source_index, first_source_order, first_error = first_invalid
            raise adapter_schema_error(
                "synplanner",
                target_id,
                (
                    f"no valid routes; skipped {invalid_count} invalid route(s), "
                    f"first invalid route source_index={first_source_index} "
                    f"source_order={first_source_order}: {first_error}"
                ),
                skipped_routes=invalid_count,
                first_invalid_source_index=first_source_index,
                first_invalid_source_order=first_source_order,
                first_validation_error=first_error,
            )

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        try:
            route_root = SynPlannerMoleculeInput.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("synplanner", target_id, "invalid molecule route root") from exc
        route_target = build_bipartite_molecule(
            route_root,
            adapter="synplanner",
            mode=mode,
            reaction_fields=_reaction_fields,
            remove_mapping=True,
        )
        if route_target is None:
            raise AdapterLogicError(
                "SynPlanner target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "synplanner", "target_id": target_id},
            )
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, remove_mapping=True)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "synplanner", target.id, expected_smiles=expected_smiles, actual_smiles=route_target.smiles
                )
        return Route(target=route_target)


# SECTION: Helpers


def _reaction_fields(node: SynPlannerReactionInput) -> dict[str, Any]:
    return {"mapped_reaction_smiles": node.smiles, "annotations": {"source_smiles": node.smiles}}
