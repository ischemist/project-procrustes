from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_tree_from_precursor_map
from retrocast.domain.chem import canonicalize_smiles
from retrocast.domain.schemas import BenchmarkTree, MoleculeNode, TargetInfo
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.utils.hashing import generate_molecule_hash
from retrocast.utils.logging import logger


class TtlReaction(BaseModel):
    product: str
    reactants: list[str]


class TtlRoute(BaseModel):
    reactions: list[TtlReaction]
    metadata: dict[str, Any] = {}


class TtlRouteList(RootModel[list[TtlRoute]]):
    root: list[TtlRoute]


class TtlRetroAdapter(BaseAdapter):
    """adapter for converting pre-processed ttlretro outputs to the benchmarktree schema."""

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates the pre-processed json data for ttlretro, transforms it, and yields benchmarktree objects.
        """
        try:
            validated_data = TtlRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - pre-processed data for target '{target_info.id}' failed schema validation. error: {e}")
            return

        for route in validated_data.root:
            try:
                tree = self._transform(route, target_info)
                yield tree
            except RetroCastException as e:
                logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
                continue

    def _transform(self, route: TtlRoute, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single ttlretro route.
        raises RetroCastException on failure.
        """
        if not route.reactions:
            retrosynthetic_tree = MoleculeNode(
                id="ursa-mol-root",
                molecule_hash=generate_molecule_hash(target_info.smiles),
                smiles=target_info.smiles,
                is_starting_material=True,
                reactions=[],
            )
            return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

        root_smiles = canonicalize_smiles(route.reactions[0].product)
        if root_smiles != target_info.smiles:
            raise AdapterLogicError(
                f"route's final product '{root_smiles}' does not match expected target '{target_info.smiles}'."
            )

        precursor_map = self._build_precursor_map(route)
        # refactor: use the common recursive builder.
        retrosynthetic_tree = build_tree_from_precursor_map(smiles=root_smiles, precursor_map=precursor_map)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

    def _build_precursor_map(self, route: TtlRoute) -> PrecursorMap:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: PrecursorMap = {}
        for reaction in route.reactions:
            canon_product = canonicalize_smiles(reaction.product)
            canon_reactants = [canonicalize_smiles(r) for r in reaction.reactants]
            precursor_map[canon_product] = canon_reactants
        return precursor_map
