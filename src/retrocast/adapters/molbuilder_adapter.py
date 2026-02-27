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
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic input models (validation layer)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
#  Adapter
# ---------------------------------------------------------------------------


class MolBuilderAdapter(BaseAdapter):
    """Adapter for converting MolBuilder retrosynthesis output to the Route schema."""

    def cast(
        self,
        raw_target_data: Any,
        target: TargetIdentity,
        ignore_stereo: bool = False,
    ) -> Generator[Route, None, None]:
        """Validate raw MolBuilder data and yield Route objects."""
        try:
            validated_routes = MolBuilderRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.debug(
                f"  - Raw data for target '{target.id}' failed MolBuilder schema validation. Error: {e}"
            )
            return

        for rank, tree_root in enumerate(validated_routes.root, start=1):
            try:
                route = self._transform(tree_root, target, rank, ignore_stereo=ignore_stereo)
                yield route
            except RetroCastException as e:
                logger.debug(f"  - Route for '{target.id}' failed transformation: {e}")
                continue

    def _transform(
        self,
        raw_root: MolBuilderNode,
        target: TargetIdentity,
        rank: int,
        ignore_stereo: bool = False,
    ) -> Route:
        """Transform a single MolBuilder tree into a Route."""
        target_molecule = self._build_molecule(raw_root, ignore_stereo=ignore_stereo)

        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if target_molecule.smiles != expected_smiles:
            msg = (
                f"Mismatched SMILES for target {target.id}. "
                f"Expected canonical: {expected_smiles}, but adapter produced: {target_molecule.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        # Collect route-level metadata from the root disconnection
        route_metadata: dict[str, Any] = {}
        if raw_root.best_disconnection is not None:
            route_metadata["score"] = raw_root.best_disconnection.score

        return Route(target=target_molecule, rank=rank, metadata=route_metadata)

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
            raise AdapterLogicError(f"cycle detected in route graph involving smiles: {canon_smiles}")

        new_visited = visited | {canon_smiles}
        is_leaf = node.is_purchasable or not bool(node.children)

        if is_leaf:
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # Recurse into children
        reactant_molecules: list[Molecule] = []
        for child in node.children:
            reactant_mol = self._build_molecule(child, visited=new_visited, ignore_stereo=ignore_stereo)
            reactant_molecules.append(reactant_mol)

        # Extract reaction metadata from best_disconnection
        step_metadata: dict[str, Any] = {}
        template: str | None = None
        if node.best_disconnection is not None:
            disc = node.best_disconnection
            step_metadata["reaction_name"] = disc.reaction_name
            step_metadata["score"] = disc.score
            if disc.named_reaction:
                step_metadata["named_reaction"] = disc.named_reaction
            if disc.category:
                step_metadata["category"] = disc.category
            # MolBuilder templates are name-based, not SMARTS
            template = disc.reaction_name

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
            metadata={},
        )
