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

# --- pydantic models for input validation ---


class SynLlamaRouteInput(BaseModel):
    synthesis_string: str


class SynLlamaRouteList(RootModel[list[SynLlamaRouteInput]]):
    pass


class SynLlaMaAdapter(BaseAdapter):
    """adapter for converting pre-processed synllama outputs to the Route schema."""

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """validates the pre-processed json data for synllama and yields Route objects."""
        try:
            validated_routes = SynLlamaRouteList.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - data for target '{target.id}' failed synllama schema validation. error: {e}")
            return

        for rank, route in enumerate(validated_routes.root, start=1):
            try:
                route_obj = self._transform(route, target, rank=rank, ignore_stereo=ignore_stereo)
                yield route_obj
            except RetroCastException as e:
                logger.warning(f"  - route for '{target.id}' failed transformation: {e}")
                continue

    def _transform(
        self, route: SynLlamaRouteInput, target: TargetIdentity, rank: int, ignore_stereo: bool = False
    ) -> Route:
        """orchestrates the transformation of a single synllama route string."""
        # the final product is always the last element in the semicolon-delimited string.
        # this is the most reliable way to identify it.
        synthesis_parts = [p.strip() for p in route.synthesis_string.split(";") if p.strip()]
        if not synthesis_parts:
            raise AdapterLogicError("synthesis string is empty.")

        # the final product is always the last element. this is the most reliable way to identify it.
        parsed_target_smiles = canonicalize_smiles(synthesis_parts[-1], ignore_stereo=ignore_stereo)
        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if parsed_target_smiles != expected_smiles:
            msg = (
                f"mismatched smiles for target {target.id}. "
                f"expected canonical: {expected_smiles}, but adapter produced: {parsed_target_smiles}"
            )
            raise AdapterLogicError(msg)

        precursor_map = self._parse_synthesis_string(route.synthesis_string, ignore_stereo=ignore_stereo)
        target_molecule = self._build_molecule_from_precursor_map(
            smiles=SmilesStr(target.smiles), precursor_map=precursor_map, ignore_stereo=ignore_stereo
        )
        return Route(target=target_molecule, rank=rank, metadata={})

    def _build_molecule_from_precursor_map(
        self,
        smiles: SmilesStr,
        precursor_map: dict[SmilesStr, list[SmilesStr]],
        visited: set[SmilesStr] | None = None,
        ignore_stereo: bool = False,
    ) -> Molecule:
        """Recursively build a Molecule tree from a precursor map."""
        if visited is None:
            visited = set()

        # Cycle detection
        if smiles in visited:
            logger.warning(f"Cycle detected for {smiles}, treating as leaf")
            return Molecule(
                smiles=smiles,
                inchikey=get_inchi_key(smiles),
                synthesis_step=None,
                metadata={},
            )

        new_visited = visited | {smiles}

        # Check if this is a leaf (not in precursor map)
        if smiles not in precursor_map:
            return Molecule(
                smiles=smiles,
                inchikey=get_inchi_key(smiles),
                synthesis_step=None,
                metadata={},
            )

        # Build reactants recursively
        reactant_molecules = []
        for reactant_smiles in precursor_map[smiles]:
            reactant_mol = self._build_molecule_from_precursor_map(
                smiles=reactant_smiles, precursor_map=precursor_map, visited=new_visited, ignore_stereo=ignore_stereo
            )
            reactant_molecules.append(reactant_mol)

        # Create the reaction step
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            metadata={},
        )

        # Create the molecule with its synthesis step
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )

    def _parse_synthesis_string(
        self, synthesis_str: str, ignore_stereo: bool = False
    ) -> dict[SmilesStr, list[SmilesStr]]:
        """
        parses a multi-step synllama route string into a precursor map.
        the format is a sequence of `reactants;template;product` chunks, chained together.
        e.g., r1;r2;t1;p1;r3;t2;p2 means p1=f(r1,r2) and p2=f(p1,r3).
        """
        precursor_map: dict[SmilesStr, list[SmilesStr]] = {}
        # clean up parts: remove whitespace and empty strings from sequences like ';;'
        parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]

        if not parts:
            raise AdapterLogicError("synthesis string is empty.")

        template_indices = [i for i, p in enumerate(parts) if p.startswith("R") and p[1:].isdigit()]

        if not template_indices:
            # if no templates, assume no reactions. it's a purchasable molecule.
            return precursor_map

        last_product_smi = None
        reactant_start_idx = 0
        for template_idx in template_indices:
            product_idx = template_idx + 1
            if product_idx >= len(parts):
                raise AdapterLogicError(f"malformed route: template '{parts[template_idx]}' has no product.")

            product_smiles = canonicalize_smiles(parts[product_idx], ignore_stereo=ignore_stereo)
            explicit_reactant_parts = parts[reactant_start_idx:template_idx]
            all_reactants = [canonicalize_smiles(r, ignore_stereo=ignore_stereo) for r in explicit_reactant_parts]
            if last_product_smi:
                all_reactants.append(last_product_smi)

            if not all_reactants:
                raise AdapterLogicError(f"no reactants found for product '{parts[product_idx]}'")

            precursor_map[product_smiles] = all_reactants

            last_product_smi = product_smiles
            reactant_start_idx = product_idx + 1

        return precursor_map
