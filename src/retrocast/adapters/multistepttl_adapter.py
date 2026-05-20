from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from retrocast._warnings import warn_deprecated
from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_route_transform_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
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


class MultiStepTTLAdapter(BaseAdapter):
    """adapter for converting pre-processed multistepttl outputs to the route schema."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Validate raw TTLRetro data and expose one route-like payload per entry.
        """
        target_id = source_key or "<unknown>"
        try:
            validated_data = TtlRouteList.model_validate(raw_data)
        except ValidationError as e:
            raise adapter_schema_error("multistepttl", target_id, "invalid pre-processed route list") from e

        for rank, route in enumerate(validated_data.root, start=1):
            yield RawRouteEntry(
                payload=route,
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
        if not isinstance(raw_route, TtlRoute):
            raw_route = TtlRoute.model_validate(raw_route)
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(
        self,
        route: TtlRoute,
        target: TargetIdentity | None,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        orchestrates the transformation of a single ttlretro route.
        raises RetroCastException on failure.
        """
        if not route.reactions:
            if target is None:
                raise adapter_route_transform_error(
                    "multistepttl",
                    "<unknown>",
                    "route does not encode target smiles for zero-reaction entries",
                )
            # no reactions means the target is already a starting material
            target_molecule = Molecule(
                smiles=SmilesStr(target.smiles),
                inchikey=get_inchi_key(target.smiles),
                synthesis_step=None,
                metadata={},
            )
            return Route(target=target_molecule, metadata={})

        root_smiles = canonicalize_smiles(route.reactions[0].product, ignore_stereo=ignore_stereo)
        if target is not None:
            expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
            if root_smiles != expected_smiles:
                raise adapter_target_mismatch(
                    "multistepttl",
                    target.id,
                    expected_smiles=expected_smiles,
                    actual_smiles=root_smiles,
                )

        # build precursor map for recursive traversal
        precursor_map = self._build_precursor_map(route, ignore_stereo=ignore_stereo)
        target_molecule = self._build_molecule(root_smiles, precursor_map, visited=set(), ignore_stereo=ignore_stereo)

        return Route(target=target_molecule, metadata=route.metadata)

    def _build_precursor_map(self, route: TtlRoute, ignore_stereo: bool = False) -> dict[str, list[str]]:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: dict[str, list[str]] = {}
        for reaction in route.reactions:
            canon_product = str(canonicalize_smiles(reaction.product, ignore_stereo=ignore_stereo))
            canon_reactants = [str(canonicalize_smiles(r, ignore_stereo=ignore_stereo)) for r in reaction.reactants]
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
            raise adapter_cycle_error("multistepttl", canon_smiles)

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


_DEPRECATED_ADAPTER_ALIASES: dict[str, type[BaseAdapter]] = {
    "TtlRetroAdapter": MultiStepTTLAdapter,
}


def __getattr__(name: str) -> Any:
    adapter_type = _DEPRECATED_ADAPTER_ALIASES.get(name)
    if adapter_type is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        old=f"{__name__}.{name}",
        new=f"{__name__}.MultiStepTTLAdapter",
        remove_in="0.7",
        stacklevel=2,
    )
    globals()[name] = adapter_type
    return adapter_type
