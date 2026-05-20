import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import adapter_cycle_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


class DMSTree(BaseModel):
    """
    A Pydantic model for the raw output from "DMS" models.

    This recursively validates the structure of a synthetic tree node,
    ensuring it has a 'smiles' string and a list of 'children' nodes.
    """

    smiles: str  # we don't canonicalize yet; this is raw input
    children: list["DMSTree"] = Field(default_factory=list)


class DMSRouteList(RootModel[list[DMSTree]]):
    """
    Represents the raw model output for a single target, which is a list of routes.
    """

    pass


class DirectMultiStepAdapter(BaseAdapter):
    """Adapter for converting DirectMultiStep-style model outputs to the Route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        try:
            validated_routes = DMSRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("dms", target_id, "invalid route list") from e

        for rank, dms_tree_root in enumerate(validated_routes.root, start=1):
            yield RawRouteEntry(
                payload=dms_tree_root,
                source_key=source_key,
                target_hint_id=None,
                target_hint_smiles=None,
                source_order=rank,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        dms_tree_root = DMSTree.model_validate(raw_route)
        target_molecule = self._build_molecule(dms_node=dms_tree_root, ignore_stereo=ignore_stereo)

        if expected_target is not None:
            expected_smiles = canonicalize_smiles(expected_target.smiles, ignore_stereo=ignore_stereo)
            if target_molecule.smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "dms",
                    expected_target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=target_molecule.smiles,
                )

        return Route(target=target_molecule, metadata={})

    def _build_molecule(
        self, dms_node: DMSTree, visited: set[SmilesStr] | None = None, ignore_stereo: bool = False
    ) -> Molecule:
        """
        Recursively builds a Molecule from a DMS tree node.
        This will propagate InvalidSmilesError if it occurs.
        """
        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(dms_node.smiles, ignore_stereo=ignore_stereo)

        if canon_smiles in visited:
            raise adapter_cycle_error("dms", canon_smiles)

        new_visited = visited | {canon_smiles}
        is_leaf = not bool(dms_node.children)

        if is_leaf:
            # This is a starting material (leaf node)
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # Build reactants recursively
        reactant_molecules: list[Molecule] = []
        for child_node in dms_node.children:
            reactant_mol = self._build_molecule(dms_node=child_node, visited=new_visited, ignore_stereo=ignore_stereo)
            reactant_molecules.append(reactant_mol)

        # Create the reaction step
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=None,
            reagents=None,
            solvents=None,
            metadata={},
        )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )

    @staticmethod
    def calculate_route_length(dms_node: DMSTree) -> int:
        """
        Calculate the length of a route from the raw DMS tree structure.

        This counts the number of reactions (steps) in the longest path
        from the target to any starting material.
        """
        if not dms_node.children:
            return 0

        max_child_length = 0
        for child in dms_node.children:
            child_length = DirectMultiStepAdapter.calculate_route_length(child)
            max_child_length = max(max_child_length, child_length)

        return max_child_length + 1


DMSAdapter = DirectMultiStepAdapter
