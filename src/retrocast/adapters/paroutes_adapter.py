from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_tree_from_bipartite_node
from retrocast.domain.schemas import BenchmarkTree, TargetInfo
from retrocast.exceptions import AdapterLogicError, RetroCastException
from retrocast.utils.logging import logger

# --- pydantic models for input validation ---
# this format is effectively identical to aizynthfinder's output,
# just with different metadata in the reaction nodes.


class PaRoutesReactionMetadata(BaseModel):
    id: str = Field(..., alias="ID")


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


class PaRoutesAdapter(BaseAdapter):
    """adapter for converting paroutes experimental routes to the benchmarktree schema."""

    _MODERN_YEAR_PATTERN = re.compile(r"^US(20\d{2})")
    _SPECIAL_PREFIX_PATTERN = re.compile(r"^US[A-Z]+")

    def __init__(self) -> None:
        """initialize the adapter with a stats counter."""
        self.year_counts: dict[str, int] = defaultdict(int)
        self.unparsed_categories: dict[str, int] = defaultdict(int)

    def _get_patent_ids(self, node: PaRoutesMoleculeInput) -> set[str]:
        """recursively traverses the raw tree to collect all unique patent ids from reaction nodes."""
        patent_ids: set[str] = set()
        for reaction_node in node.children:
            try:
                # the patent id is the part before the first semicolon
                patent_id = reaction_node.metadata.id.split(";")[0]
                patent_ids.add(patent_id)
            except (IndexError, AttributeError):
                logger.warning(f"could not parse patent id from metadata: {reaction_node.metadata}")

            for reactant_node in reaction_node.children:
                patent_ids.update(self._get_patent_ids(reactant_node))
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

    def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        """
        validates a single paroutes route, checks for patent consistency, and transforms it.
        """
        try:
            # unlike other adapters, the raw data for one target is a single route object, not a list.
            validated_route_root = PaRoutesMoleculeInput.model_validate(raw_target_data)
        except ValidationError as e:
            logger.warning(f"  - raw data for target '{target_info.id}' failed paroutes schema validation. error: {e}")
            return

        # --- custom validation: ensure all reactions are from the same patent ---
        patent_ids = self._get_patent_ids(validated_route_root)
        if len(patent_ids) > 1:
            logger.warning(
                f"  - skipping route for '{target_info.id}': contains reactions from multiple patents: {patent_ids}"
            )
            return
        elif len(patent_ids) == 1:
            patent_id = list(patent_ids)[0]
            year = self._get_year_from_patent_id(patent_id)
            if year:
                self.year_counts[year] += 1

        if not patent_ids:  # skip if no patent id was found
            return

        try:
            tree = self._transform(validated_route_root, target_info)
            yield tree
        except RetroCastException as e:
            logger.warning(f"  - route for '{target_info.id}' failed transformation: {e}")
            return

    def _transform(self, paroutes_root: PaRoutesMoleculeInput, target_info: TargetInfo) -> BenchmarkTree:
        """
        orchestrates the transformation of a single validated paroutes tree.
        """
        # this format is a standard bipartite graph, so we reuse the common builder.
        retrosynthetic_tree = build_tree_from_bipartite_node(
            raw_mol_node=paroutes_root, path_prefix="retrocast-mol-root"
        )

        if retrosynthetic_tree.smiles != target_info.smiles:
            msg = (
                f"mismatched smiles for target {target_info.id}. "
                f"expected canonical: {target_info.smiles}, but adapter produced: {retrosynthetic_tree.smiles}"
            )
            logger.error(msg)
            raise AdapterLogicError(msg)

        return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)

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
