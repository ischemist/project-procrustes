from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)


class TtlReaction(BaseModel):
    product: str
    reactants: list[str]


class TtlRoute(BaseModel):
    reactions: list[TtlReaction]
    metadata: dict[str, Any] = {}


class TtlRouteList(RootModel[list[TtlRoute]]):
    root: list[TtlRoute]


class TtlRetroAdapter(BaseAdapter):
    """adapter for converting pre-processed ttlretro outputs to the route schema."""

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        validates the pre-processed json data for ttlretro, transforms it, and yields route objects.
        """
        try:
            validated_data = TtlRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - pre-processed data for target '{target.id}' failed schema validation. error: {e}")
            return

        for rank, route in enumerate(validated_data.root, start=1):
            try:
                adapted_route = self._transform(route, target, rank, ignore_stereo=ignore_stereo)
                yield adapted_route
            except RetroCastException as e:
                logger.warning(f"  - route for '{target.id}' failed transformation: {e}")
                continue

    def _transform(self, route: TtlRoute, target: TargetIdentity, rank: int, ignore_stereo: bool = False) -> Route:
        """
        orchestrates the transformation of a single ttlretro route.
        raises RetroCastException on failure.
        """
        if not route.reactions:
            # no reactions means the target is already a starting material
            target_molecule = Molecule(
                smiles=SmilesStr(target.smiles),
                inchikey=get_inchi_key(target.smiles),
                synthesis_step=None,
                metadata={},
            )
            return Route(target=target_molecule, rank=rank, metadata={})

        root_smiles = canonicalize_smiles(route.reactions[0].product, ignore_stereo=ignore_stereo)
        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if root_smiles != expected_smiles:
            raise AdapterLogicError(
                f"route's final product '{root_smiles}' does not match expected target '{expected_smiles}'."
            )

        # build precursor map for recursive traversal
        precursor_map = self._build_precursor_map(route, ignore_stereo=ignore_stereo)
        target_molecule = self._build_molecule(root_smiles, precursor_map, visited=set(), ignore_stereo=ignore_stereo)

        return Route(target=target_molecule, rank=rank, metadata=route.metadata)

    def _build_precursor_map(self, route: TtlRoute, ignore_stereo: bool = False) -> dict[str, list[str]]:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: dict[str, list[str]] = {}
        for reaction in route.reactions:
            canon_product = canonicalize_smiles(reaction.product, ignore_stereo=ignore_stereo)
            canon_reactants = [canonicalize_smiles(r, ignore_stereo=ignore_stereo) for r in reaction.reactants]
            precursor_map[canon_product] = canon_reactants
        return precursor_map

    def _build_molecule(
        self, smiles: str, precursor_map: dict[str, list[str]], visited: set[str], ignore_stereo: bool = False
    ) -> Molecule:
        """
        recursively builds a molecule object from the precursor map.
        raises AdapterLogicError if a cycle is detected.
        """
        canon_smiles = canonicalize_smiles(smiles, ignore_stereo=ignore_stereo)

        if canon_smiles in visited:
            raise AdapterLogicError(f"cycle detected: molecule '{canon_smiles}' appears multiple times in route.")

        # if the molecule is not in the precursor map, it's a starting material
        if canon_smiles not in precursor_map:
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # mark this molecule as visited
        visited.add(canon_smiles)

        # recursively build reactant molecules
        reactant_smiles_list = precursor_map[canon_smiles]
        reactants = [
            self._build_molecule(r_smiles, precursor_map, visited.copy(), ignore_stereo=ignore_stereo)
            for r_smiles in reactant_smiles_list
        ]

        synthesis_step = ReactionStep(reactants=reactants, metadata={})

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )
