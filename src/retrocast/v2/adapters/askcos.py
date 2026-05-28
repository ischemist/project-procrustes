from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_missing_node_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError, UnsupportedAdapterFeatureError
from retrocast.typing import ReactionSmilesStr, SmilesStr
from retrocast.v2.adapters.base import AdaptMode, RawRouteEntry
from retrocast.v2.models.route import Molecule, Reaction, Route
from retrocast.v2.models.task import Target

logger = logging.getLogger(__name__)

ASKCOS_ROOT_UUID = "00000000-0000-0000-0000-000000000000"


class AskcosBaseNode(BaseModel):
    smiles: str
    id: str


class AskcosChemicalNode(AskcosBaseNode):
    type: Literal["chemical"]
    terminal: bool


class AskcosModelMetadata(BaseModel):
    source: dict[str, Any] = Field(default_factory=dict)

    def get_template(self) -> str | None:
        template = self.source.get("template")
        if not isinstance(template, dict):
            return None
        reaction_smarts = template.get("reaction_smarts")
        return reaction_smarts if isinstance(reaction_smarts, str) else None


class AskcosReactionProperties(BaseModel):
    mapped_smiles: str | None = None


class AskcosReactionNode(AskcosBaseNode):
    type: Literal["reaction"]
    reaction_properties: AskcosReactionProperties | None = None
    model_metadata: list[AskcosModelMetadata] = Field(default_factory=list)


AskcosNode = Annotated[AskcosChemicalNode | AskcosReactionNode, Field(discriminator="type")]


class AskcosPathwayEdge(BaseModel):
    source: str
    target: str


class AskcosUDS(BaseModel):
    node_dict: dict[str, AskcosNode]
    uuid2smiles: dict[str, str]
    pathways: list[list[AskcosPathwayEdge]]


class AskcosResults(BaseModel):
    uds: AskcosUDS

    stats: dict[str, Any] = Field(default_factory=dict)


class AskcosOutput(BaseModel):
    results: AskcosResults


@dataclass(frozen=True, slots=True)
class AskcosPathwayPayload:
    pathway_edges: list[AskcosPathwayEdge]
    uuid2smiles: dict[str, str]
    node_dict: dict[str, AskcosNode]
    annotations: dict[str, Any]


class AskcosAdapter:
    def __init__(self, use_full_graph: bool = False) -> None:
        self.use_full_graph = use_full_graph

    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        if self.use_full_graph:
            raise UnsupportedAdapterFeatureError(
                "ASKCOS full-graph route extraction is not implemented",
                context={"adapter": "askcos", "feature": "full_graph"},
            )

        target_id = source_key or "<unknown>"
        try:
            output = AskcosOutput.model_validate(raw_payload)
        except ValidationError as exc:
            raise adapter_schema_error("askcos", target_id, "invalid output") from exc

        annotations = _extract_run_annotations(output.results.stats)
        uds = output.results.uds
        for pathway_index, pathway_edges in enumerate(uds.pathways, start=1):
            yield RawRouteEntry(
                payload=AskcosPathwayPayload(
                    pathway_edges=pathway_edges,
                    uuid2smiles=uds.uuid2smiles,
                    node_dict=uds.node_dict,
                    annotations=annotations,
                ),
                source_key=source_key,
                source_order=pathway_index,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route:
        target_id = target.id if target is not None else "<unknown>"
        if not isinstance(raw_route, AskcosPathwayPayload):
            raise adapter_schema_error("askcos", target_id, "invalid pathway")

        route_target = self._build_target_molecule(raw_route, mode=mode)
        if route_target is None:
            raise AdapterLogicError(
                "ASKCOS target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": "askcos", "target_id": target_id},
            )

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles)
            if route_target.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "askcos",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=route_target.smiles,
                )

        return Route(target=route_target, annotations=raw_route.annotations)

    def _build_target_molecule(self, raw_route: AskcosPathwayPayload, *, mode: AdaptMode) -> Molecule | None:
        adj_list: dict[str, list[str]] = defaultdict(list)
        for edge in raw_route.pathway_edges:
            adj_list[edge.source].append(edge.target)

        if ASKCOS_ROOT_UUID not in raw_route.uuid2smiles:
            raise adapter_missing_node_error(
                "askcos", node_id=ASKCOS_ROOT_UUID, lookup="uuid2smiles", role="root chemical"
            )

        return self._build_molecule(
            chem_uuid=ASKCOS_ROOT_UUID,
            adj_list=adj_list,
            uuid2smiles=raw_route.uuid2smiles,
            node_dict=raw_route.node_dict,
            visited=set(),
            mode=mode,
        )

    def _build_molecule(
        self,
        *,
        chem_uuid: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        visited: set[SmilesStr],
        mode: AdaptMode,
    ) -> Molecule | None:
        raw_smiles = uuid2smiles.get(chem_uuid)
        if not raw_smiles:
            raise adapter_missing_node_error("askcos", node_id=chem_uuid, lookup="uuid2smiles", role="chemical")

        node = node_dict.get(raw_smiles)
        if not isinstance(node, AskcosChemicalNode):
            raise adapter_missing_node_error("askcos", node_id=raw_smiles, lookup="node_dict", role="chemical")

        try:
            canon_smiles = canonicalize_smiles(node.smiles)
        except InvalidSmilesError:
            if mode == "prune":
                return None
            raise

        if canon_smiles in visited:
            raise adapter_cycle_error("askcos", canon_smiles)

        if node.terminal or chem_uuid not in adj_list:
            return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles))

        child_reaction_uuids = adj_list[chem_uuid]
        if len(child_reaction_uuids) > 1:
            logger.warning("molecule %s has multiple child reactions; only the first is used", canon_smiles)

        reaction = self._build_reaction(
            rxn_uuid=child_reaction_uuids[0],
            adj_list=adj_list,
            uuid2smiles=uuid2smiles,
            node_dict=node_dict,
            visited=visited | {canon_smiles},
            mode=mode,
        )
        if reaction is None:
            if mode == "prune":
                return None
            raise AdapterLogicError(
                "ASKCOS reaction has no reactants",
                code="adapter.reaction_empty",
                context={"adapter": "askcos", "smiles": canon_smiles},
            )

        return Molecule(smiles=canon_smiles, inchikey=get_inchi_key(canon_smiles), product_of=reaction)

    def _build_reaction(
        self,
        *,
        rxn_uuid: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        visited: set[SmilesStr],
        mode: AdaptMode,
    ) -> Reaction | None:
        raw_smiles = uuid2smiles.get(rxn_uuid)
        if not raw_smiles:
            raise adapter_missing_node_error("askcos", node_id=rxn_uuid, lookup="uuid2smiles", role="reaction")

        node = node_dict.get(raw_smiles)
        if not isinstance(node, AskcosReactionNode):
            raise adapter_missing_node_error("askcos", node_id=raw_smiles, lookup="node_dict", role="reaction")

        reactants: list[Molecule] = []
        for reactant_uuid in adj_list.get(rxn_uuid, []):
            reactant = self._build_molecule(
                chem_uuid=reactant_uuid,
                adj_list=adj_list,
                uuid2smiles=uuid2smiles,
                node_dict=node_dict,
                visited=visited,
                mode=mode,
            )
            if reactant is not None:
                reactants.append(reactant)

        if not reactants:
            return None

        mapped_reaction_smiles = None
        if node.reaction_properties and node.reaction_properties.mapped_smiles:
            mapped_reaction_smiles = ReactionSmilesStr(node.reaction_properties.mapped_smiles)

        return Reaction(
            reactants=reactants,
            mapped_reaction_smiles=mapped_reaction_smiles,
            template=node.model_metadata[0].get_template() if node.model_metadata else None,
            annotations={"source_id": node.id},
        )


def _extract_run_annotations(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in ("total_iterations", "total_chemicals", "total_reactions", "total_templates", "total_paths")
        if key in stats
    }
