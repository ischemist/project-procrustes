from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import BenchmarkTree, MoleculeNode, ReactionNode, TargetInfo
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.typing import ReactionSmilesStr, SmilesStr
from retrocast.utils.hashing import generate_molecule_hash
from retrocast.utils.logging import logger

# --- pydantic models for input validation ---


class AskcosBaseNode(BaseModel):
    smiles: str
    id: str


class AskcosChemicalNode(AskcosBaseNode):
    type: Literal["chemical"]
    terminal: bool


class AskcosReactionNode(AskcosBaseNode):
    type: Literal["reaction"]


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

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """validates raw askcos data, transforms its pathways, and yields benchmarktree objects."""
        if self.use_full_graph:
            raise NotImplementedError("extracting routes from the full askcos search graph is not yet implemented.")

        try:
            validated_output = AskcosOutput.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target_info.id}' failed askcos schema validation. error: {e}")
            return

        uds = validated_output.results.uds

        for i, pathway_edges in enumerate(uds.pathways):
            try:
                tree = self._transform_pathway(
                    pathway_edges=pathway_edges,
                    uuid2smiles=uds.uuid2smiles,
                    node_dict=uds.node_dict,
                    target_info=target_info,
                )
                yield tree
            except RetroCastException as e:
                logger.warning(f"  - pathway {i} for target '{target_info.id}' failed transformation: {e}")
                continue

    def _transform_pathway(
        self,
        pathway_edges: list[AskcosPathwayEdge],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
        target_info: TargetInfo,
    ) -> BenchmarkTree:
        """transforms a single askcos pathway (represented by its edges) into a benchmarktree."""
        adj_list = defaultdict(list)
        for edge in pathway_edges:
            adj_list[edge.source].append(edge.target)

        root_uuid = "00000000-0000-0000-0000-000000000000"
        if root_uuid not in uuid2smiles:
            raise AdapterLogicError("root uuid not found in pathway data.")

        retrosynthetic_tree = self._build_molecule_node(
            chem_uuid=root_uuid,
            path_prefix="retrocast-mol-root",
            adj_list=adj_list,
            uuid2smiles=uuid2smiles,
            node_dict=node_dict,
        )

        if retrosynthetic_tree.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {retrosynthetic_tree.smiles}"
            )
            raise AdapterLogicError(msg)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _build_molecule_node(
        self,
        chem_uuid: str,
        path_prefix: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
    ) -> MoleculeNode:
        """recursively builds a canonical moleculenode from a chemical uuid."""
        raw_smiles = uuid2smiles.get(chem_uuid)
        if not raw_smiles:
            raise AdapterLogicError(f"uuid '{chem_uuid}' not found in uuid2smiles map.")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosChemicalNode):
            raise AdapterLogicError(f"node data for smiles '{raw_smiles}' not found or not a chemical node.")

        canon_smiles = canonicalize_smiles(node_data.smiles)
        is_starting_mat = node_data.terminal
        reactions = []

        if not is_starting_mat and chem_uuid in adj_list:
            child_reaction_uuids = adj_list[chem_uuid]
            if len(child_reaction_uuids) > 1:
                logger.warning(f"molecule {canon_smiles} has multiple child reactions in pathway; using first one.")

            rxn_uuid = child_reaction_uuids[0]
            reactions.append(
                self._build_reaction_node(
                    rxn_uuid=rxn_uuid,
                    product_smiles=canon_smiles,
                    path_prefix=path_prefix,
                    adj_list=adj_list,
                    uuid2smiles=uuid2smiles,
                    node_dict=node_dict,
                )
            )

        return MoleculeNode(
            id=path_prefix,
            molecule_hash=generate_molecule_hash(canon_smiles),
            smiles=canon_smiles,
            is_starting_material=is_starting_mat,
            reactions=reactions,
        )

    def _build_reaction_node(
        self,
        rxn_uuid: str,
        product_smiles: SmilesStr,
        path_prefix: str,
        adj_list: dict[str, list[str]],
        uuid2smiles: dict[str, str],
        node_dict: dict[str, AskcosNode],
    ) -> ReactionNode:
        """builds a canonical reactionnode from a reaction uuid."""
        raw_smiles = uuid2smiles.get(rxn_uuid)
        if not raw_smiles:
            raise AdapterLogicError(f"uuid '{rxn_uuid}' not found in uuid2smiles map.")

        node_data = node_dict.get(raw_smiles)
        if not node_data or not isinstance(node_data, AskcosReactionNode):
            raise AdapterLogicError(f"node data for reaction '{raw_smiles}' not found or not a reaction node.")

        reactants: list[MoleculeNode] = []
        reactant_smiles_list: list[SmilesStr] = []

        reactant_uuids = adj_list.get(rxn_uuid, [])
        for i, reactant_uuid in enumerate(reactant_uuids):
            reactant_node = self._build_molecule_node(
                chem_uuid=reactant_uuid,
                path_prefix=f"{path_prefix}-{i}",
                adj_list=adj_list,
                uuid2smiles=uuid2smiles,
                node_dict=node_dict,
            )
            reactants.append(reactant_node)
            reactant_smiles_list.append(reactant_node.smiles)

        reaction_smiles = ReactionSmilesStr(f"{'.'.join(sorted(reactant_smiles_list))}>>{product_smiles}")

        return ReactionNode(
            id=path_prefix.replace("retrocast-mol", "retrocast-rxn"),
            reaction_smiles=reaction_smiles,
            reactants=reactants,
        )
