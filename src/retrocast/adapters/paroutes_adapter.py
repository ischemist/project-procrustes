from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.errors import (
    adapter_cycle_error,
    adapter_node_type_error,
    adapter_schema_error,
    adapter_target_mismatch,
)
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import AdapterError, AdapterLogicError, ChemError, RetroCastException
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity
from retrocast.typing import ReactionSmilesStr, SmilesStr

logger = logging.getLogger(__name__)

# --- pydantic models for input validation ---
# this format is effectively identical to aizynthfinder's output,
# just with different metadata in the reaction nodes.


class PaRoutesReactionMetadata(BaseModel):
    id: str = Field(..., alias="ID")
    rsmi: str | None = None  # reaction-mapped SMILES
    smiles: str | None = None  # mapped reactants>>product without the condition slot
    reaction_hash: str | None = None
    ring_breaker: bool | None = Field(None, alias="RingBreaker")


class PaRoutesBaseNode(BaseModel):
    smiles: str
    children: list[PaRoutesNode] = Field(default_factory=list)


class PaRoutesMoleculeInput(PaRoutesBaseNode):
    type: Literal["mol"]
    in_stock: bool = False


class PaRoutesReactionInput(PaRoutesBaseNode):
    type: Literal["reaction"]
    metadata: PaRoutesReactionMetadata
    children: list[PaRoutesMoleculeInput] = Field(default_factory=list)


PaRoutesNode = Annotated[PaRoutesMoleculeInput | PaRoutesReactionInput, Field(discriminator="type")]

# pydantic needs this to resolve the forward references in the recursive models
PaRoutesMoleculeInput.model_rebuild()
PaRoutesReactionInput.model_rebuild()


class ConditionSlotParseStatistics(BaseModel):
    malformed_rsmi_count: int = 0
    uncanonicalizable_token_count: int = 0
    uncanonicalizable_tokens: dict[str, int] = Field(default_factory=lambda: defaultdict(int))

    @property
    def distinct_uncanonicalizable_token_count(self) -> int:
        return len(self.uncanonicalizable_tokens)

    @property
    def top_uncanonicalizable_tokens(self) -> list[tuple[str, int]]:
        return sorted(self.uncanonicalizable_tokens.items(), key=lambda item: (-item[1], item[0]))[:5]

    def to_manifest_dict(self) -> dict[str, int]:
        return {
            "malformed_rsmi_count": self.malformed_rsmi_count,
            "uncanonicalizable_token_count": self.uncanonicalizable_token_count,
            "distinct_uncanonicalizable_token_count": self.distinct_uncanonicalizable_token_count,
        }


@dataclass(frozen=True)
class PaRoutesAdaptationResult:
    route: Route | None
    error: RetroCastException | None = None


class PaRoutesAdapter(BaseAdapter):
    """adapter for converting paroutes experimental routes to the route schema."""

    _MODERN_YEAR_PATTERN = re.compile(r"^US(20\d{2})")
    _SPECIAL_PREFIX_PATTERN = re.compile(r"^US[A-Z]+")

    def __init__(self) -> None:
        """initialize the adapter with a stats counter."""
        self.year_counts: dict[str, int] = defaultdict(int)
        self.unparsed_categories: dict[str, int] = defaultdict(int)

    def _get_patent_ids(self, node: PaRoutesMoleculeInput, visited: set[str] | None = None) -> set[str]:
        """
        recursively traverses the raw tree to collect all unique patent ids from reaction nodes.

        Args:
            node: The molecule node to traverse
            visited: Set of SMILES already visited (for cycle detection)

        Returns:
            Set of unique patent IDs found in the tree
        """
        if visited is None:
            visited = set()

        # Use raw SMILES for cycle detection (before canonicalization)
        if node.smiles in visited:
            logger.warning(f"cycle detected in _get_patent_ids for smiles: {node.smiles}")
            return set()

        new_visited = visited | {node.smiles}
        patent_ids: set[str] = set()

        for reaction_node in node.children:
            # Type guard: reaction nodes should have metadata
            if not isinstance(reaction_node, PaRoutesReactionInput):
                continue

            try:
                # the patent id is the part before the first semicolon
                patent_id = reaction_node.metadata.id.split(";")[0]
                patent_ids.add(patent_id)
            except (IndexError, AttributeError):
                logger.warning(f"could not parse patent id from metadata: {reaction_node.metadata}")

            for reactant_node in reaction_node.children:
                patent_ids.update(self._get_patent_ids(reactant_node, visited=new_visited))
        return patent_ids

    def _get_year_from_patent_id(self, patent_id: str) -> str | None:
        """extracts the year from a patent id string, or categorizes it."""
        # handles modern format: US<YYYY><serial>A1, e.g., US2015...
        match = self._MODERN_YEAR_PATTERN.match(patent_id)
        if match:
            return match.group(1)

        # handles special administrative patents like reissues (USRE...) or SIRs (USH...)
        if self._SPECIAL_PREFIX_PATTERN.match(patent_id):
            self.unparsed_categories["special/admin"] += 1
            return None

        # if it starts with US and a digit but didn't match the modern year format,
        # it's a pre-2001 granted patent. the number does not contain a year.
        if patent_id.startswith("US") and len(patent_id) > 2 and patent_id[2].isdigit():
            self.unparsed_categories["pre-2001_grant"] += 1
            return None

        self.unparsed_categories["unknown_format"] += 1
        return None

    def _extract_condition_slot(
        self,
        rsmi: str | None,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> str | None:
        """extract the raw middle condition slot from a paroutes reaction smiles string."""
        if not rsmi:
            return None

        parts = rsmi.split(">")
        if len(parts) != 3:
            condition_slot_parse_statistics.malformed_rsmi_count += 1
            return None

        condition_slot = parts[1].strip()
        return condition_slot or None

    def _parse_condition_slot_smiles(
        self,
        condition_slot: str,
        *,
        ignore_stereo: bool,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> list[SmilesStr]:
        """best-effort canonicalization of molecules listed in the condition slot."""
        parsed_smiles: list[SmilesStr] = []
        for token in condition_slot.split("."):
            token = token.strip()
            if not token:
                continue
            try:
                parsed_smiles.append(
                    canonicalize_smiles(
                        token,
                        remove_mapping=True,
                        ignore_stereo=ignore_stereo,
                    )
                )
            except RetroCastException:
                condition_slot_parse_statistics.uncanonicalizable_token_count += 1
                condition_slot_parse_statistics.uncanonicalizable_tokens[token] += 1

        return sorted(parsed_smiles)

    def _build_reaction_metadata(
        self,
        rxn_metadata: PaRoutesReactionMetadata | None,
        *,
        ignore_stereo: bool,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> dict[str, Any]:
        """build trustworthy reaction metadata from ambiguous paroutes side information."""
        if rxn_metadata is None:
            return {}

        metadata: dict[str, Any] = {
            "source_id": rxn_metadata.id,
        }
        if rxn_metadata.ring_breaker is not None:
            metadata["ring_breaker"] = rxn_metadata.ring_breaker

        condition_slot = self._extract_condition_slot(rxn_metadata.rsmi, condition_slot_parse_statistics)
        if condition_slot is not None:
            metadata["condition_slot"] = condition_slot
            condition_slot_smiles = self._parse_condition_slot_smiles(
                condition_slot,
                ignore_stereo=ignore_stereo,
                condition_slot_parse_statistics=condition_slot_parse_statistics,
            )
            if condition_slot_smiles:
                metadata["condition_slot_smiles"] = condition_slot_smiles

        return metadata

    def cast(
        self,
        raw_target_data: Any,
        target: TargetIdentity,
        ignore_stereo: bool = False,
        condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
    ) -> Generator[Route, None, None]:
        """
        validates a single paroutes route, checks for patent consistency, and transforms it.
        """
        try:
            # unlike other adapters, the raw data for one target is a single route object, not a list.
            validated_route_root = PaRoutesMoleculeInput.model_validate(raw_target_data)
        except ValidationError as e:
            raise adapter_schema_error("paroutes", target.id, "invalid molecule route root") from e

        # --- custom validation: ensure all reactions are from the same patent ---
        patent_ids = self._get_patent_ids(validated_route_root)
        if len(patent_ids) > 1:
            raise AdapterLogicError(
                f"PaRoutes route for target '{target.id}' contains reactions from multiple patents",
                code="adapter.multiple_patents",
                context={"adapter": "paroutes", "target_id": target.id, "patent_ids": sorted(patent_ids)},
            )
        elif len(patent_ids) == 1:
            patent_id = list(patent_ids)[0]
            year = self._get_year_from_patent_id(patent_id)
            if year:
                self.year_counts[year] += 1

        if not patent_ids:  # skip if no patent id was found
            raise AdapterLogicError(
                f"PaRoutes route for target '{target.id}' does not contain a patent id",
                code="adapter.patent_id_missing",
                context={"adapter": "paroutes", "target_id": target.id},
            )

        stats = condition_slot_parse_statistics or ConditionSlotParseStatistics()
        yield self._transform(
            validated_route_root,
            target,
            patent_id=list(patent_ids)[0],
            ignore_stereo=ignore_stereo,
            condition_slot_parse_statistics=stats,
        )

    def adapt_route_with_diagnostics(
        self,
        raw_route: Any,
        target: TargetIdentity,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> PaRoutesAdaptationResult:
        try:
            route = next(
                self.cast(
                    raw_route,
                    target,
                    condition_slot_parse_statistics=condition_slot_parse_statistics,
                ),
                None,
            )
            return PaRoutesAdaptationResult(route=route)
        except (AdapterError, ChemError) as exc:
            return PaRoutesAdaptationResult(route=None, error=exc)

    def _transform(
        self,
        paroutes_root: PaRoutesMoleculeInput,
        target: TargetIdentity,
        patent_id: str,
        *,
        ignore_stereo: bool = False,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> Route:
        """
        orchestrates the transformation of a single validated paroutes tree.
        """
        # build the molecule tree recursively with cycle detection
        target_molecule = self._build_molecule(
            paroutes_root,
            visited=set(),
            ignore_stereo=ignore_stereo,
            condition_slot_parse_statistics=condition_slot_parse_statistics,
        )

        expected_smiles = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if target_molecule.smiles != expected_smiles:
            raise adapter_target_mismatch(
                "paroutes",
                target.id,
                expected_smiles=expected_smiles,
                actual_smiles=target_molecule.smiles,
            )

        # add patent ID to route metadata (everything up to first semicolon)
        route_metadata = {"patent_id": patent_id}

        return Route(target=target_molecule, rank=1, metadata=route_metadata)

    def _build_molecule(
        self,
        raw_mol_node: PaRoutesMoleculeInput,
        visited: set[SmilesStr] | None = None,
        *,
        ignore_stereo: bool = False,
        condition_slot_parse_statistics: ConditionSlotParseStatistics,
    ) -> Molecule:
        """
        recursively builds a molecule from a paroutes bipartite graph node.

        Args:
            raw_mol_node: The raw molecule node from paroutes data
            visited: Set of canonical SMILES already visited (for cycle detection)
            ignore_stereo: If True, stereochemistry is stripped during SMILES canonicalization.

        Raises:
            AdapterLogicError: If a cycle is detected in the route graph
        """
        if raw_mol_node.type != "mol":
            raise adapter_node_type_error("paroutes", expected="mol", actual=raw_mol_node.type, role="molecule")

        if visited is None:
            visited = set()

        canon_smiles = canonicalize_smiles(raw_mol_node.smiles, ignore_stereo=ignore_stereo)

        # Cycle detection: check if we've seen this molecule before in the current path
        if canon_smiles in visited:
            raise adapter_cycle_error("paroutes", canon_smiles)

        # Create new visited set with current molecule added
        new_visited = visited | {canon_smiles}

        is_leaf = raw_mol_node.in_stock or not bool(raw_mol_node.children)

        if is_leaf:
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={},
            )

        # in a valid tree, a molecule has at most one reaction leading to it
        if len(raw_mol_node.children) > 1:
            logger.warning(
                f"molecule {canon_smiles} has multiple child reactions in raw output; only the first is used in a tree."
            )

        first_child = raw_mol_node.children[0]
        if not isinstance(first_child, PaRoutesReactionInput):
            actual = getattr(first_child, "type", type(first_child).__name__)
            raise adapter_node_type_error("paroutes", expected="reaction", actual=actual, role="molecule child")
        raw_reaction_node: PaRoutesReactionInput = first_child

        # build reactants recursively with updated visited set
        reactant_molecules: list[Molecule] = []
        for reactant_mol_input in raw_reaction_node.children:
            reactant_mol = self._build_molecule(
                reactant_mol_input,
                visited=new_visited,
                ignore_stereo=ignore_stereo,
                condition_slot_parse_statistics=condition_slot_parse_statistics,
            )
            reactant_molecules.append(reactant_mol)

        # extract mapped smiles (rsmi) from metadata
        rxn_metadata = raw_reaction_node.metadata
        mapped_smiles_str = rxn_metadata.rsmi if rxn_metadata else None
        mapped_smiles = ReactionSmilesStr(mapped_smiles_str) if mapped_smiles_str else None
        metadata_dict = self._build_reaction_metadata(
            rxn_metadata,
            ignore_stereo=ignore_stereo,
            condition_slot_parse_statistics=condition_slot_parse_statistics,
        )
        synthesis_step = ReactionStep(
            reactants=reactant_molecules,
            mapped_smiles=mapped_smiles,
            template=None,
            reagents=None,
            solvents=None,
            metadata=metadata_dict,
        )

        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={},
        )

    def report_statistics(self) -> None:
        """logs the collected patent year statistics."""
        if not self.year_counts and not self.unparsed_categories:
            return

        logger.info("--- PaRoutes Patent Year Statistics ---")
        if self.year_counts:
            for year, count in sorted(self.year_counts.items()):
                logger.info(f"  - Parsed Year {year}: {count} routes")
        if self.unparsed_categories:
            for category, count in sorted(self.unparsed_categories.items()):
                logger.info(f"  - Category '{category}': {count} routes")
        logger.info("-" * 39)
