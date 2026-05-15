from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_missing_node_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import UnsupportedAdapterFeatureError
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import ReactionSmilesStr, SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---


class AskcosBaseNode(BaseModel):
    smiles: str
    id: str


class AskcosChemicalNode(AskcosBaseNode):
    type: Literal["chemical"]
    terminal: bool


class AskcosTemplateSource(BaseModel):
    """Nested structure for template information."""

    reaction_smarts: str | None = None


class AskcosModelMetadata(BaseModel):
    """Model metadata containing template information."""

    source: dict[str, Any] = Field(default_factory=dict)

    def get_template(self) -> str | None:
        """Extract reaction_smarts from nested template structure."""
        template_dict = self.source.get("template", {})
        return template_dict.get("reaction_smarts") if isinstance(template_dict, dict) else None


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


class AskcosOutput(BaseModel):
    results: AskcosResults


@dataclass(frozen=True, slots=True)
class AskcosPathwayPayload:
    pathway_edges: list[AskcosPathwayEdge]
    uuid2smiles: dict[str, str]
    node_dict: dict[str, AskcosNode]
    metadata: dict[str, Any]


class AskcosAdapter(BaseAdapter):
    """adapter for converting askcos outputs to the route schema."""

    def __init__(self, use_full_graph: bool = False):
        """
        initializes the adapter.

        args:
            use_full_graph: if true, attempts to extract all possible routes
                from the full search graph instead of using the pre-computed
                pathways. defaults to false.
        """
        self.use_full_graph = use_full_graph

    def iter_raw_entries(
        self,
        raw_data: Any,
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
            validated_output = AskcosOutput.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("askcos", target_id, "invalid output") from e

        stats = raw_data.get("results", {}).get("stats", {}) if isinstance(raw_data, dict) else {}
        metadata = {
            "total_iterations": stats.get("total_iterations"),
            "total_chemicals": stats.get("total_chemicals"),
            "total_reactions": stats.get("total_reactions"),
            "total_templates": stats.get("total_templates"),
            "total_paths": stats.get("total_paths"),
        }
        uds = validated_output.results.uds

        for pathway_index, pathway_edges in enumerate(uds.pathways, start=1):
            yield RawRouteEntry(
                payload=AskcosPathwayPayload(
                    pathway_edges=pathway_edges,
                    uuid2smiles=uds.uuid2smiles,
                    node_dict=uds.node_dict,
                    metadata=metadata,
                ),
                source_key=source_key,
                target_hint_id=None,
                target_hint_smiles=None,
                source_order=pathway_index,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, AskcosPathwayPayload):
            raise adapter_schema_error(
                "askcos", expected_target.id if expected_target else "<unknown>", "invalid pathway"
            )

        target_molecule = self._build_target_molecule(
            pathway_edges=raw_route.pathway_edges,
            uuid2smiles=raw_route.uuid2smiles,
            node_dict=raw_route.node_dict,
            ignore_stereo=ignore_stereo,
        )

        if expected_target is not None:
            expected_smiles = canonicalize_smiles(expected_target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "askcos",
                    expected_target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        return Route(target=target_molecule, metadata=raw_route.metadata)

    def _build_target_molecule(
        self,
        pathway_edges: list[AskcosPathwayEdge],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        ignore_stereo: bool = False,
    ) -> Molecule:
        """build the root molecule for a single askcos pathway payload."""
        adj_list = defaultdict(list)
        for edge in pathway_edges:
            adj_list[edge.source].append(edge.target)

        root_uuid = "00000000-0000-0000-0000-000000000000"
        if root_uuid not in uuid2smiles:
            raise adapter_missing_node_error("askcos", node_id=root_uuid, lookup="uuid2smiles", role="root chemical")

        return self._build_molecule(
            chem_uuid=root_uuid,
            path_prefix="retrocast-mol-root",
            adj_list=adj_list,
            uuid2smiles=uuid2smiles,
            node_dict=node_dict,
            visited=set(),
            ignore_stereo=ignore_stereo,
        )

    def _build_molecule(
        self,
        chem_uuid: str,
        path_prefix: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        visited: set[SmilesStr] | None = None,
        ignore_stereo: bool = False,
    ) -> Molecule:
        """recursively builds a canonical molecule from a chemical uuid."""
        if visited is None:
            visited = set()

        raw_smiles = uuid2smiles.get(chem_uuid)
        if not raw_smiles:
            raise adapter_missing_node_error("askcos", node_id=chem_uuid, lookup="uuid2smiles", role="chemical")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosChemicalNode):
            raise adapter_missing_node_error("askcos", node_id=raw_smiles, lookup="node_dict", role="chemical")

        canon_smiles = canonicalize_smiles(node_data.smiles, ignore_stereo=ignore_stereo)
        if canon_smiles in visited:
            raise adapter_cycle_error("askcos", canon_smiles)

        new_visited = visited | {canon_smiles}
        is_leaf = node_data.terminal
        synthesis_step = None

        if not is_leaf and chem_uuid in adj_list:
            child_reaction_uuids = adj_list[chem_uuid]
            if len(child_reaction_uuids) > 1:
                logger.warning(f"molecule {canon_smiles} has multiple child reactions in pathway; using first one.")

            rxn_uuid = child_reaction_uuids[0]
            synthesis_step = self._build_reaction_step(
                rxn_uuid=rxn_uuid,
                product_smiles=canon_smiles,
                path_prefix=path_prefix,
                adj_list=adj_list,
                uuid2smiles=uuid2smiles,
                node_dict=node_dict,
                visited=new_visited,
                ignore_stereo=ignore_stereo,
            )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
        )

    def _build_reaction_step(
        self,
        rxn_uuid: str,
        product_smiles: SmilesStr,
        path_prefix: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        visited: set[SmilesStr],
        ignore_stereo: bool = False,
    ) -> ReactionStep:
        """builds a canonical reaction step from a reaction uuid."""
        raw_smiles = uuid2smiles.get(rxn_uuid)
        if not raw_smiles:
            raise adapter_missing_node_error("askcos", node_id=rxn_uuid, lookup="uuid2smiles", role="reaction")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosReactionNode):
            raise adapter_missing_node_error("askcos", node_id=raw_smiles, lookup="node_dict", role="reaction")

        reactants: list[Molecule] = []
        reactant_smiles_list: list[SmilesStr] = []

        reactant_uuids = adj_list.get(rxn_uuid, [])
        for i, reactant_uuid in enumerate(reactant_uuids):
            reactant_molecule = self._build_molecule(
                chem_uuid=reactant_uuid,
                path_prefix=f"{path_prefix}-{i}",
                adj_list=adj_list,
                uuid2smiles=uuid2smiles,
                node_dict=node_dict,
                visited=visited,
                ignore_stereo=ignore_stereo,
            )
            reactants.append(reactant_molecule)
            reactant_smiles_list.append(reactant_molecule.smiles)

        # Extract mapped_smiles from reaction_properties if available
        mapped_smiles = None
        if node_data.reaction_properties and node_data.reaction_properties.mapped_smiles:
            mapped_smiles = ReactionSmilesStr(node_data.reaction_properties.mapped_smiles)

        # Extract template from model_metadata if available
        template = None
        if node_data.model_metadata and len(node_data.model_metadata) > 0:
            template = node_data.model_metadata[0].get_template()

        return ReactionStep(
            reactants=reactants,
            mapped_smiles=mapped_smiles,
            template=template,
        )
