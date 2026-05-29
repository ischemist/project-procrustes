from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_node_type_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, ChemError, InvalidSmilesError
from retrocast.models.route import Molecule, Reaction, Route
from retrocast.models.task import Target
from retrocast.typing import ReactionSmilesStr, SmilesStr

logger = logging.getLogger(__name__)


# SECTION: Raw PaRoutes Schema


class PaRoutesReactionMetadata(BaseModel):
    id: str = Field(..., alias="ID")
    rsmi: str | None = None
    smiles: str | None = None
    reaction_hash: str | None = None
    ring_breaker: bool | None = Field(None, alias="RingBreaker")


class PaRoutesBaseNode(BaseModel):
    smiles: str
    children: list[PaRoutesNode] = Field(default_factory=list)


class PaRoutesMoleculeInput(PaRoutesBaseNode):
    type: Literal["mol"]
    in_stock: bool = False


class PaRoutesReactionInput(PaRoutesBaseNode):
    type: Literal["reaction"]
    metadata: PaRoutesReactionMetadata
    children: list[PaRoutesMoleculeInput] = Field(default_factory=list)


PaRoutesNode = Annotated[PaRoutesMoleculeInput | PaRoutesReactionInput, Field(discriminator="type")]


# SECTION: Condition Slot Annotations


class ConditionSlotParseStatistics(BaseModel):
    malformed_rsmi_count: int = 0
    uncanonicalizable_token_count: int = 0
    uncanonicalizable_tokens: defaultdict[str, int] = Field(default_factory=lambda: defaultdict(int))

    @property
    def distinct_uncanonicalizable_token_count(self) -> int:
        return len(self.uncanonicalizable_tokens)

    @property
    def top_uncanonicalizable_tokens(self) -> list[tuple[str, int]]:
        return sorted(self.uncanonicalizable_tokens.items(), key=lambda item: (-item[1], item[0]))[:5]


def _extract_condition_slot(
    rsmi: str | None,
    *,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> str | None:
    if not rsmi:
        return None

    parts = rsmi.split(">")
    if len(parts) != 3:
        if condition_slot_parse_statistics is not None:
            condition_slot_parse_statistics.malformed_rsmi_count += 1
        return None

    condition_slot = parts[1].strip()
    return condition_slot or None


def _parse_condition_slot_smiles(
    condition_slot: str,
    *,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> list[SmilesStr]:
    parsed_smiles: list[SmilesStr] = []
    for token in condition_slot.split("."):
        token = token.strip()
        if not token:
            continue
        try:
            parsed_smiles.append(canonicalize_smiles(token, remove_mapping=True))
        except ChemError:
            if condition_slot_parse_statistics is not None:
                condition_slot_parse_statistics.uncanonicalizable_token_count += 1
                condition_slot_parse_statistics.uncanonicalizable_tokens[token] += 1

    return sorted(parsed_smiles)


def _build_condition_slot_annotations(
    *,
    source_id: str,
    rsmi: str | None,
    ring_breaker: bool | None,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> dict[str, Any]:
    annotations: dict[str, Any] = {"source_id": source_id}
    if ring_breaker is not None:
        annotations["ring_breaker"] = ring_breaker

    condition_slot = _extract_condition_slot(
        rsmi,
        condition_slot_parse_statistics=condition_slot_parse_statistics,
    )
    if condition_slot is not None:
        annotations["condition_slot"] = condition_slot
        condition_slot_smiles = _parse_condition_slot_smiles(
            condition_slot,
            condition_slot_parse_statistics=condition_slot_parse_statistics,
        )
        if condition_slot_smiles:
            annotations["condition_slot_smiles"] = condition_slot_smiles

    return annotations


# SECTION: Adapter


class PaRoutesAdapter:
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        if not isinstance(raw_payload, Mapping):
            target_id = source_key or "<unknown>"
            raise adapter_schema_error("paroutes", target_id, "expected route root or mapping of target ids to routes")

        is_route_root = raw_payload.get("type") == "mol" or "smiles" in raw_payload or "children" in raw_payload
        if is_route_root:
            target_id = source_key or "<unknown>"
            yield RawRouteEntry(
                payload=self._validate_route_root(raw_payload, target_id=target_id),
                source_key=source_key,
                source_order=1,
            )
            return

        for source_order, (target_id, raw_route) in enumerate(raw_payload.items(), start=1):
            if not isinstance(target_id, str):
                raise adapter_schema_error("paroutes", source_key or "<unknown>", "target id keys must be strings")
            yield RawRouteEntry(
                payload=self._validate_route_root(raw_route, target_id=target_id),
                source_key=target_id,
                source_order=source_order,
            )

    def _validate_route_root(self, raw_route: Any, *, target_id: str) -> PaRoutesMoleculeInput:
        try:
            return PaRoutesMoleculeInput.model_validate(raw_route)
        except ValidationError as exc:
            raise adapter_schema_error("paroutes", target_id, "invalid molecule route root") from exc

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        raw_route = self._validate_route_root(raw_route, target_id=target_id)
        patent_ids = self._get_patent_ids(raw_route, mode=mode)
        if not patent_ids:
            raise AdapterLogicError(
                f"PaRoutes route for target '{target_id}' does not contain a patent id",
                code="adapter.patent_id_missing",
                context={"adapter": "paroutes", "target_id": target_id},
            )
        if len(patent_ids) > 1:
            raise AdapterLogicError(
                f"PaRoutes route for target '{target_id}' contains reactions from multiple patents",
                code="adapter.multiple_patents",
                context={"adapter": "paroutes", "target_id": target_id, "patent_ids": sorted(patent_ids)},
            )

        route_target = self._build_molecule(raw_route, visited=set(), mode=mode)
        if route_target is None:
            raise AdapterLogicError(
                "PaRoutes target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "paroutes", "target_id": target_id},
            )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "paroutes",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=route_target.smiles,
                )

        return Route(target=route_target, annotations={"patent_id": next(iter(patent_ids))})

    def _get_patent_ids(
        self,
        node: PaRoutesMoleculeInput,
        *,
        mode: AdaptMode,
        visited: set[str] | None = None,
    ) -> set[str]:
        if visited is None:
            visited = set()
        try:
            canon_smiles = canonicalize_smiles(node.smiles)
        except InvalidSmilesError:
            if mode == "prune":
                return set()
            raise
        if canon_smiles in visited:
            raise adapter_cycle_error("paroutes", canon_smiles)

        patent_ids: set[str] = set()
        new_visited = visited | {canon_smiles}
        if node.children:
            reaction_node = self._require_reaction_node(node.children[0], role="molecule child")
            patent_id = reaction_node.metadata.id.split(";", 1)[0].strip()
            if not patent_id:
                raise AdapterLogicError(
                    "PaRoutes reaction metadata contains an empty patent id",
                    code="adapter.patent_id_missing",
                    context={"adapter": "paroutes", "source_id": reaction_node.metadata.id},
                )
            patent_ids.add(patent_id)
            for reactant_node in reaction_node.children:
                patent_ids.update(self._get_patent_ids(reactant_node, mode=mode, visited=new_visited))
        return patent_ids

    def _require_reaction_node(self, node: PaRoutesNode, *, role: str) -> PaRoutesReactionInput:
        if isinstance(node, PaRoutesReactionInput):
            return node
        raise adapter_node_type_error("paroutes", expected="reaction", actual=node.type, role=role)

    def _build_molecule(
        self,
        raw_mol_node: PaRoutesMoleculeInput,
        visited: set[SmilesStr],
        *,
        mode: AdaptMode,
    ) -> Molecule | None:
        try:
            canon_smiles = canonicalize_smiles(raw_mol_node.smiles)
        except InvalidSmilesError:
            if mode == "prune":
                return None
            raise

        if canon_smiles in visited:
            raise adapter_cycle_error("paroutes", canon_smiles)

        if raw_mol_node.in_stock or not raw_mol_node.children:
            return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))

        if len(raw_mol_node.children) > 1:
            logger.warning("molecule %s has multiple child reactions; only the first is used", canon_smiles)

        reaction_node = self._require_reaction_node(raw_mol_node.children[0], role="molecule child")

        reactants = []
        for reactant_node in reaction_node.children:
            reactant = self._build_molecule(reactant_node, visited | {canon_smiles}, mode=mode)
            if reactant is not None:
                reactants.append(reactant)
        if not reactants:
            if mode == "prune":
                return None
            raise AdapterLogicError(
                "PaRoutes reaction has no reactants",
                code="adapter.reaction_empty",
                context={"adapter": "paroutes", "smiles": canon_smiles},
            )

        reaction = Reaction(
            reactants=reactants,
            mapped_reaction_smiles=ReactionSmilesStr(reaction_node.metadata.rsmi)
            if reaction_node.metadata.rsmi
            else None,
            annotations=_build_condition_slot_annotations(
                source_id=reaction_node.metadata.id,
                rsmi=reaction_node.metadata.rsmi,
                ring_breaker=reaction_node.metadata.ring_breaker,
            ),
        )
        return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles), product_of=reaction)


# SECTION: Diagnostics


def analyze_condition_slots(
    raw_route: Mapping[str, Any],
    *,
    stats: ConditionSlotParseStatistics,
) -> None:
    def visit(node: Mapping[str, Any]) -> None:
        children = node.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            if not isinstance(child, Mapping):
                continue
            if child.get("type") == "reaction":
                metadata = child.get("metadata")
                if isinstance(metadata, Mapping) and isinstance(metadata.get("ID"), str):
                    _build_condition_slot_annotations(
                        source_id=metadata["ID"],
                        rsmi=metadata.get("rsmi") if isinstance(metadata.get("rsmi"), str) else None,
                        ring_breaker=metadata.get("RingBreaker")
                        if isinstance(metadata.get("RingBreaker"), bool)
                        else None,
                        condition_slot_parse_statistics=stats,
                    )
            visit(child)

    visit(raw_route)
