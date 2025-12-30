from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---


class RetrochimeraReaction(BaseModel):
    reactants: list[str]
    product: str
    probability: float
    metadata: dict[str, Any] = {}


class RetrochimeraRoute(BaseModel):
    reactions: list[RetrochimeraReaction]
    num_steps: int
    step_probability_min: float
    step_probability_product: float


class RetrochimeraOutput(BaseModel):
    routes: list[RetrochimeraRoute]
    num_routes: int
    num_routes_initial_extraction: int = 0
    target_is_purchasable: bool = False
    num_model_calls_total: int = 0
    num_model_calls_new: int = 0
    num_model_calls_cached: int = 0
    num_nodes_explored: int = 0
    time_taken_s_search: float = 0.0
    time_taken_s_extraction: float = 0.0


class RetrochimeraResult(BaseModel):
    request: dict[str, Any] | None = None
    outputs: list[RetrochimeraOutput] | None = None
    error: dict[str, Any] | None = None
    time_taken_s: float = 0.0


class RetrochimeraData(BaseModel):
    smiles: str
    result: RetrochimeraResult


class RetrochimeraAdapter(BaseAdapter):
    """adapter for converting retrochimera-style outputs to the Route schema."""

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        validates raw retrochimera data, transforms it, and yields Route objects.
        """
        try:
            validated_data = RetrochimeraData.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target.id}' failed retrochimera schema validation. error: {e}")
            return

        if validated_data.result.error is not None:
            error_msg = validated_data.result.error.get("message", "unknown error")
            error_type = validated_data.result.error.get("type", "unknown")
            logger.warning(f"  - retrochimera reported an error for target '{target.id}': {error_type} - {error_msg}")
            return

        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if canonicalize_smiles(validated_data.smiles, ignore_stereo=ignore_stereo) != expected_smiles:
            logger.warning(
                f"  - mismatched smiles for target '{target.id}': expected {expected_smiles}, got {canonicalize_smiles(validated_data.smiles, ignore_stereo=ignore_stereo)}"
            )
            return

        if validated_data.result.outputs is None:
            logger.warning(f"  - no outputs found for target '{target.id}'")
            return

        rank = 1
        for output in validated_data.result.outputs:
            for route in output.routes:
                try:
                    route_obj = self._transform(route, target, rank=rank, ignore_stereo=ignore_stereo)
                    yield route_obj
                    rank += 1
                except RetroCastException as e:
                    logger.warning(f"  - route for '{target.id}' failed transformation: {e}")
                    continue

    def _transform(
        self, route: RetrochimeraRoute, target: TargetIdentity, rank: int, ignore_stereo: bool = False
    ) -> Route:
        """
        orchestrates the transformation of a single retrochimera route.
        raises RetroCastException on failure.
        """
        precursor_map = self._build_precursor_map(route, ignore_stereo=ignore_stereo)
        target_molecule = self._build_molecule_from_precursor_map(
            smiles=SmilesStr(target.smiles),
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
        )

        return Route(target=target_molecule, rank=rank, metadata={})

    def _build_precursor_map(
        self, route: RetrochimeraRoute, ignore_stereo: bool = False
    ) -> dict[SmilesStr, list[SmilesStr]]:
        """
        builds a precursor map from the route's reactions.
        each product maps to its list of reactant smiles.
        """
        precursor_map: dict[SmilesStr, list[SmilesStr]] = {}
        for reaction in route.reactions:
            canon_product = canonicalize_smiles(reaction.product, ignore_stereo=ignore_stereo)
            canon_reactants = [canonicalize_smiles(r, ignore_stereo=ignore_stereo) for r in reaction.reactants]
            precursor_map[canon_product] = canon_reactants
        return precursor_map

    def _build_molecule_from_precursor_map(
        self,
        smiles: SmilesStr,
        precursor_map: dict[SmilesStr, list[SmilesStr]],
        visited: set[SmilesStr] | None = None,
        ignore_stereo: bool = False,
    ) -> Molecule:
        """
        recursively builds a Molecule from a precursor map, with cycle detection.
        """
        if visited is None:
            visited = set()

        # Cycle detection
        if smiles in visited:
            logger.warning(f"Cycle detected in route graph involving smiles: {smiles}. Treating as a leaf node.")
            return Molecule(
                smiles=smiles,
                inchikey=get_inchi_key(smiles),
                synthesis_step=None,
                metadata={},
            )

        new_visited = visited | {smiles}
        is_leaf = smiles not in precursor_map

        if is_leaf:
            # This is a starting material (leaf node)
            return Molecule(
                smiles=smiles,
                inchikey=get_inchi_key(smiles),
                synthesis_step=None,
                metadata={},
            )

        # Build reactants recursively
        reactant_smiles_list = precursor_map[smiles]
        reactant_molecules: list[Molecule] = []

        for reactant_smi in reactant_smiles_list:
            reactant_mol = self._build_molecule_from_precursor_map(
                smiles=reactant_smi,
                precursor_map=precursor_map,
                visited=new_visited,
                ignore_stereo=ignore_stereo,
            )
            reactant_molecules.append(reactant_mol)

        # Create the reaction step
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=None,
            template=None,
            reagents=None,
            solvents=None,
            metadata={},
        )

        # Create the molecule with its synthesis step
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )
