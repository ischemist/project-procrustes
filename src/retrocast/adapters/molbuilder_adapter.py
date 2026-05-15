"""Adapter for MolBuilder retrosynthesis output.

MolBuilder (https://pypi.org/project/molbuilder/) is a process chemistry
toolkit that produces retrosynthesis trees via template-based disconnection.
Its output is a recursive tree where each node has SMILES, purchasability,
and an optional best_disconnection with reaction metadata.

Raw data format (route-centric):
    A list of tree roots, one per alternative route.  Each tree is a
    recursive node with the following shape::

        {
            "smiles": "CCO",
            "is_purchasable": false,
            "best_disconnection": {
                "reaction_name": "NaBH4 Reduction",
                "named_reaction": "NaBH4 Reduction",
                "category": "reduction",
                "score": 0.85,
                "precursors": [
                    {"smiles": "CC=O", "name": "acetaldehyde", "cost_per_kg": 15.0}
                ]
            },
            "children": [
                {"smiles": "CC=O", "is_purchasable": true, "children": []}
            ]
        }

    Leaf nodes have ``is_purchasable: true`` and empty ``children``.
"""

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


class MolBuilderPrecursor(BaseModel):
    """A precursor entry from a MolBuilder disconnection."""

    smiles: str
    name: str = ""
    cost_per_kg: float = 0.0


class MolBuilderDisconnection(BaseModel):
    """Reaction metadata from a MolBuilder disconnection."""

    reaction_name: str
    named_reaction: str | None = None
    category: str = ""
    score: float = 0.0
    precursors: list[MolBuilderPrecursor] = Field(default_factory=list)


class MolBuilderNode(BaseModel):
    """A node in MolBuilder's retrosynthesis tree."""

    smiles: str
    is_purchasable: bool = False
    functional_groups: list[str] = Field(default_factory=list)
    best_disconnection: MolBuilderDisconnection | None = None
    children: list["MolBuilderNode"] = Field(default_factory=list)


class MolBuilderRouteList(RootModel[list[MolBuilderNode]]):
    """Top-level validation: a list of MolBuilder tree roots."""

    pass


class MolBuilderAdapter(BaseAdapter):
    """Adapter for converting MolBuilder retrosynthesis output to the Route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
        expected_target: TargetIdentity | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = expected_target.id if expected_target is not None else source_key or "<unknown>"
        try:
            validated_routes = MolBuilderRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("molbuilder", target_id, "invalid route list") from e

        for rank, tree_root in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=tree_root,
                source_key=source_key,
                expected_target_id=expected_target.id if expected_target is not None else None,
                expected_target_smiles=expected_target.smiles if expected_target is not None else None,
                source_order=rank,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, MolBuilderNode):
            raw_route = MolBuilderNode.model_validate(raw_route)
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(
        self,
        raw_root: MolBuilderNode,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
    ) -> Route:
        """Transform a single MolBuilder tree into a Route."""
        target_molecule = self._build_molecule(raw_root, ignore_stereo=ignore_stereo)

        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "molbuilder",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        route_metadata: dict[str, Any] = {}
        if raw_root.best_disconnection is not None:
            route_metadata["score"] = raw_root.best_disconnection.score

        return Route(target=target_molecule, metadata=route_metadata)

    def _build_molecule(
        self,
        node: MolBuilderNode,
        visited: set[SmilesStr] | None = None,
        ignore_stereo: bool = False,
    ) -> Molecule:
        """Recursively build a Molecule from a MolBuilder tree node."""
        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(node.smiles, ignore_stereo=ignore_stereo)

        if canon_smiles in visited:
            raise adapter_cycle_error("molbuilder", canon_smiles)

        visited.add(canon_smiles)
        is_leaf = node.is_purchasable or not bool(node.children)

        mol_metadata: dict[str, Any] = {}
        if node.functional_groups:
            mol_metadata["functional_groups"] = node.functional_groups

        if is_leaf:
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata=mol_metadata,
            )

        reactant_molecules: list[Molecule] = []
        for child in node.children:
            reactant_mol = self._build_molecule(child, visited=set(visited), ignore_stereo=ignore_stereo)
            reactant_molecules.append(reactant_mol)

        disc = node.best_disconnection
        step_metadata: dict[str, Any] = {}
        template = None
        if disc is not None:
            reaction_name = disc.reaction_name.strip()
            if reaction_name:
                step_metadata["reaction_name"] = reaction_name
                template = reaction_name
            step_metadata["score"] = disc.score
            if disc.named_reaction:
                step_metadata["named_reaction"] = disc.named_reaction
            if disc.category:
                step_metadata["category"] = disc.category

        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=None,
            template=template,
            reagents=None,
            solvents=None,
            metadata=step_metadata,
        )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata=mol_metadata,
        )
