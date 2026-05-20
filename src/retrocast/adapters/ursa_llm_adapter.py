from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_transform_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SYNTHESIS_STEP_RE = re.compile(r"<synthesis_step>(.*?)</synthesis_step>", re.DOTALL)
_PRODUCT_RE = re.compile(r"<product>(.*?)</product>", re.DOTALL)
_REACTANT_RE = re.compile(r"<reactant>(.*?)</reactant>", re.DOTALL)
_SMILES_RE = re.compile(r"<smiles>(.*?)</smiles>", re.DOTALL)
_SM_TOKEN_RE = re.compile(r"<sm_([^>]+)>")


def _extract_smiles(block: str) -> str | None:
    """Extract a SMILES from a `<product>` or `<reactant>` block."""
    match = _SMILES_RE.search(block)
    if not match:
        return None
    inner = match.group(1).strip()
    if "<sm_" in inner:
        tokens = _SM_TOKEN_RE.findall(inner)
        return "".join(tokens) if tokens else None
    return inner if inner else None


def _extract_completion_text(raw_record: Any) -> str | None:
    if not isinstance(raw_record, dict):
        return None
    completion = raw_record.get("completion")
    return completion if isinstance(completion, str) else None


def _extract_source_target_smiles(raw_record: Any) -> str | None:
    if not isinstance(raw_record, dict):
        return None

    meta = raw_record.get("meta")
    if not isinstance(meta, dict):
        return None

    product_smiles = meta.get("product_smiles")
    if not isinstance(product_smiles, str):
        return None

    return product_smiles


class UrsaAdapter(BaseAdapter):
    """Adapter for Ursa LLM completions containing `<synthesis_step>` XML blocks."""

    adapter_key = "ursa-llm"

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        if not isinstance(raw_data, list):
            raise adapter_schema_error(self.adapter_key, target_id, "expected a list of completion records")

        for row_index, raw_record in enumerate(raw_data, start=1):
            completion = _extract_completion_text(raw_record)
            if completion is None:
                raise adapter_schema_error(
                    self.adapter_key,
                    target_id,
                    "each completion record must be a dict with a string 'completion' field",
                    record_index=row_index - 1,
                )
            source_target_smiles = _extract_source_target_smiles(raw_record)
            target_hint_smiles: str | None = None
            if source_target_smiles is None:
                if source_key is None:
                    raise adapter_schema_error(
                        self.adapter_key,
                        target_id,
                        "completion records must include meta.product_smiles",
                        record_index=row_index - 1,
                    )
            else:
                try:
                    target_hint_smiles = canonicalize_smiles(source_target_smiles)
                except RetroCastException as exc:
                    raise adapter_schema_error(
                        self.adapter_key,
                        target_id,
                        "completion record has invalid meta.product_smiles",
                        record_index=row_index - 1,
                    ) from exc
            yield RawRouteEntry(
                payload=completion,
                source_key=source_key,
                source_row_index=row_index,
                target_hint_id=None,
                target_hint_smiles=target_hint_smiles,
                source_order=row_index,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        if not isinstance(raw_route, str):
            raise adapter_schema_error(
                self.adapter_key,
                expected_target.id if expected_target is not None else "<unknown>",
                "expected completion text",
            )
        if expected_target is None:
            raise adapter_route_transform_error(
                self.adapter_key,
                "<unknown>",
                "ursa llm adaptation requires an expected target",
            )
        return self._transform(raw_route, expected_target, ignore_stereo=ignore_stereo)

    def _transform(self, completion: str, target: TargetIdentity, ignore_stereo: bool = False) -> Route:
        precursor_map = self._parse_completion(completion, ignore_stereo=ignore_stereo)
        if not precursor_map:
            raise adapter_route_transform_error(self.adapter_key, target.id, "no synthesis steps found in completion")

        expected_target = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if expected_target not in precursor_map:
            raise adapter_target_mismatch(
                self.adapter_key,
                target.id,
                expected_smiles=expected_target,
                actual_smiles=f"missing:{expected_target}",
            )

        molecule = build_molecule_from_precursor_map(
            smiles=SmilesStr(expected_target),
            precursor_map=precursor_map,
            ignore_stereo=ignore_stereo,
            adapter=self.adapter_key,
        )
        return Route(target=molecule, metadata={})

    def _parse_completion(self, completion: str, ignore_stereo: bool = False) -> PrecursorMap:
        """Parse `<synthesis_step>` blocks from a completion into a precursor map."""
        cleaned = _THINK_BLOCK_RE.sub("", completion)
        precursor_map: PrecursorMap = {}

        for step_idx, step_text in enumerate(_SYNTHESIS_STEP_RE.findall(cleaned)):
            product_match = _PRODUCT_RE.search(step_text)
            if not product_match:
                logger.debug(f"step {step_idx}: missing <product> block, skipping")
                continue

            product_raw = _extract_smiles(product_match.group(1))
            if not product_raw:
                logger.debug(f"step {step_idx}: missing or empty <smiles> in <product>, skipping")
                continue

            reactant_smiles: list[SmilesStr] = []
            for reactant_block in _REACTANT_RE.findall(step_text):
                reactant_raw = _extract_smiles(reactant_block)
                if not reactant_raw:
                    continue
                try:
                    reactant_smiles.append(canonicalize_smiles(reactant_raw, ignore_stereo=ignore_stereo))
                except RetroCastException:
                    logger.debug(f"step {step_idx}: invalid reactant smiles '{reactant_raw}', skipping reactant")
                    continue

            if not reactant_smiles:
                logger.debug(f"step {step_idx}: no valid reactants, skipping step")
                continue

            try:
                product_canon = canonicalize_smiles(product_raw, ignore_stereo=ignore_stereo)
            except RetroCastException:
                logger.debug(f"step {step_idx}: invalid product smiles '{product_raw}', skipping step")
                continue

            precursor_map[product_canon] = reactant_smiles

        return precursor_map


UrsaLlmAdapter = UrsaAdapter


__all__ = [
    "UrsaAdapter",
    "UrsaLlmAdapter",
]
