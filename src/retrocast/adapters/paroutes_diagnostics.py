from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import ChemError
from retrocast.typing import SmilesStr

_MODERN_YEAR_PATTERN = re.compile(r"^US(20\d{2})")
_SPECIAL_PREFIX_PATTERN = re.compile(r"^US[A-Z]+")


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


class PatentIdParseStatistics(BaseModel):
    year_counts: dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    unparsed_categories: dict[str, int] = Field(default_factory=lambda: defaultdict(int))

    def record_patent_id(self, patent_id: str) -> str | None:
        year, category = classify_patent_id(patent_id)
        if year is not None:
            self.year_counts[year] += 1
        elif category is not None:
            self.unparsed_categories[category] += 1
        return year

    def to_manifest_dict(self) -> dict[str, dict[str, int]]:
        return {
            "year_counts": dict(sorted(self.year_counts.items())),
            "unparsed_categories": dict(sorted(self.unparsed_categories.items())),
        }


def classify_patent_id(patent_id: str) -> tuple[str | None, str | None]:
    """Extract a modern patent year or categorize the patent id format."""
    match = _MODERN_YEAR_PATTERN.match(patent_id)
    if match:
        return match.group(1), None

    if _SPECIAL_PREFIX_PATTERN.match(patent_id):
        return None, "special/admin"

    if patent_id.startswith("US") and len(patent_id) > 2 and patent_id[2].isdigit():
        return None, "pre-2001_grant"

    return None, "unknown_format"


def log_patent_id_parse_statistics(stats: PatentIdParseStatistics, *, logger_name: str) -> None:
    """Log per-run patent-id parsing summaries when they exist."""
    if not stats.year_counts and not stats.unparsed_categories:
        return

    from logging import getLogger

    logger = getLogger(logger_name)
    logger.info("--- PaRoutes Patent Year Statistics ---")
    for year, count in sorted(stats.year_counts.items()):
        logger.info("  - Parsed Year %s: %s routes", year, count)
    for category, count in sorted(stats.unparsed_categories.items()):
        logger.info("  - Category '%s': %s routes", category, count)
    logger.info("---------------------------------------")


def _extract_condition_slot(
    rsmi: str | None,
    *,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> str | None:
    if not rsmi:
        return None

    parts = rsmi.split(">")
    if len(parts) != 3:
        if condition_slot_parse_statistics is not None:
            condition_slot_parse_statistics.malformed_rsmi_count += 1
        return None

    condition_slot = parts[1].strip()
    return condition_slot or None


def _parse_condition_slot_smiles(
    condition_slot: str,
    *,
    ignore_stereo: bool,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> list[SmilesStr]:
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
        except ChemError:
            if condition_slot_parse_statistics is not None:
                condition_slot_parse_statistics.uncanonicalizable_token_count += 1
                condition_slot_parse_statistics.uncanonicalizable_tokens[token] += 1

    return sorted(parsed_smiles)


def build_condition_slot_metadata(
    *,
    source_id: str,
    rsmi: str | None,
    ring_breaker: bool | None,
    ignore_stereo: bool,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
) -> dict[str, Any]:
    """Build trustworthy reaction metadata from PaRoutes side information."""
    metadata: dict[str, Any] = {"source_id": source_id}
    if ring_breaker is not None:
        metadata["ring_breaker"] = ring_breaker

    condition_slot = _extract_condition_slot(
        rsmi,
        condition_slot_parse_statistics=condition_slot_parse_statistics,
    )
    if condition_slot is not None:
        metadata["condition_slot"] = condition_slot
        condition_slot_smiles = _parse_condition_slot_smiles(
            condition_slot,
            ignore_stereo=ignore_stereo,
            condition_slot_parse_statistics=condition_slot_parse_statistics,
        )
        if condition_slot_smiles:
            metadata["condition_slot_smiles"] = condition_slot_smiles

    return metadata


def collect_raw_paroutes_route_diagnostics(
    raw_route: Mapping[str, Any],
    *,
    patent_id_parse_statistics: PatentIdParseStatistics | None = None,
    condition_slot_parse_statistics: ConditionSlotParseStatistics | None = None,
    ignore_stereo: bool = False,
) -> None:
    """Collect non-fatal raw diagnostics from a PaRoutes route payload."""

    def _visit(node: Mapping[str, Any]) -> None:
        children = node.get("children")
        if not isinstance(children, list):
            return

        for child in children:
            if not isinstance(child, Mapping):
                continue

            if child.get("type") == "reaction":
                metadata = child.get("metadata")
                if isinstance(metadata, Mapping):
                    source_id = metadata.get("ID")
                    rsmi = metadata.get("rsmi")
                    ring_breaker = metadata.get("RingBreaker")

                    if patent_id_parse_statistics is not None and isinstance(source_id, str):
                        patent_id_parse_statistics.record_patent_id(source_id.split(";")[0])

                    if condition_slot_parse_statistics is not None and isinstance(source_id, str):
                        build_condition_slot_metadata(
                            source_id=source_id,
                            rsmi=rsmi if isinstance(rsmi, str) else None,
                            ring_breaker=ring_breaker if isinstance(ring_breaker, bool) else None,
                            ignore_stereo=ignore_stereo,
                            condition_slot_parse_statistics=condition_slot_parse_statistics,
                        )

            _visit(child)

    _visit(raw_route)
