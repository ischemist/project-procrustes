from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, RetroCastException
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


class AskcosAdapter(BaseAdapter):
    """adapter for converting askcos outputs to the benchmarktree schema."""

    def __init__(self, use_full_graph: bool = False):
        """
        initializes the adapter.

        args:
            use_full_graph: if true, attempts to extract all possible routes
                from the full search graph instead of using the pre-computed
                pathways. defaults to false.
        """
        self.use_full_graph = use_full_graph

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """validates raw askcos data, transforms its pathways, and yields route objects."""
        if self.use_full_graph:
            raise NotImplementedError("extracting routes from the full askcos search graph is not yet implemented.")

        try:
            validated_output = AskcosOutput.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target.id}' failed askcos schema validation. error: {e}")
            return

        uds = validated_output.results.uds

        # Extract metadata from stats if available
        stats = raw_target_data.get("results", {}).get("stats", {})
        metadata = {
            "total_iterations": stats.get("total_iterations"),
            "total_chemicals": stats.get("total_chemicals"),
            "total_reactions": stats.get("total_reactions"),
            "total_templates": stats.get("total_templates"),
            "total_paths": stats.get("total_paths"),
        }

        for i, pathway_edges in enumerate(uds.pathways):
            try:
                route = self._transform_pathway(
                    pathway_edges=pathway_edges,
                    uuid2smiles=uds.uuid2smiles,
                    node_dict=uds.node_dict,
                    target_input=target,
                    rank=i + 1,
                    metadata=metadata,
                    ignore_stereo=ignore_stereo,
                )
                yield route
            except RetroCastException as e:
                logger.warning(f"  - pathway {i} for target '{target.id}' failed transformation: {e}")
                continue

    def _transform_pathway(
        self,
        pathway_edges: list[AskcosPathwayEdge],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        target_input: TargetIdentity,
        rank: int,
        metadata: dict[str, Any],
        ignore_stereo: bool = False,
    ) -> Route:
        """transforms a single askcos pathway (represented by its edges) into a route."""
        adj_list = defaultdict(list)
        for edge in pathway_edges:
            adj_list[edge.source].append(edge.target)

        root_uuid = "00000000-0000-0000-0000-000000000000"
        if root_uuid not in uuid2smiles:
            raise AdapterLogicError("root uuid not found in pathway data.")

        target_molecule = self._build_molecule(
            chem_uuid=root_uuid,
            path_prefix="retrocast-mol-root",
            adj_list=adj_list,
            uuid2smiles=uuid2smiles,
            node_dict=node_dict,
            ignore_stereo=ignore_stereo,
        )

        if target_molecule.smiles != target_input.smiles:
            msg = (
                f"mismatched smiles for target {target_input.id}. "
                f"expected canonical: {target_input.smiles}, but adapter produced: {target_molecule.smiles}"
            )
            raise AdapterLogicError(msg)

        return Route(target=target_molecule, rank=rank, metadata=metadata)

    def _build_molecule(
        self,
        chem_uuid: str,
        path_prefix: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        ignore_stereo: bool = False,
    ) -> Molecule:
        """recursively builds a canonical molecule from a chemical uuid."""
        raw_smiles = uuid2smiles.get(chem_uuid)
        if not raw_smiles:
            raise AdapterLogicError(f"uuid '{chem_uuid}' not found in uuid2smiles map.")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosChemicalNode):
            raise AdapterLogicError(f"node data for smiles '{raw_smiles}' not found or not a chemical node.")

        canon_smiles = canonicalize_smiles(node_data.smiles, ignore_stereo=ignore_stereo)
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
        ignore_stereo: bool = False,
    ) -> ReactionStep:
        """builds a canonical reaction step from a reaction uuid."""
        raw_smiles = uuid2smiles.get(rxn_uuid)
        if not raw_smiles:
            raise AdapterLogicError(f"uuid '{rxn_uuid}' not found in uuid2smiles map.")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosReactionNode):
            raise AdapterLogicError(f"node data for reaction '{raw_smiles}' not found or not a reaction node.")

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
