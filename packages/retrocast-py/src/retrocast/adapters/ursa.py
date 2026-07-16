from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, cast

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_transform_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError, InvalidSmilesError, RetroCastException
from retrocast.models.route import Route
from retrocast.models.task import Target
from retrocast.typing import SmilesStr

# SECTION: URSA Completion Parsing

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SYNTHESIS_STEP_RE = re.compile(r"<synthesis_step>(.*?)</synthesis_step>", re.DOTALL)
_PRODUCT_RE = re.compile(r"<product>(.*?)</product>", re.DOTALL)
_REACTANT_RE = re.compile(r"<reactant>(.*?)</reactant>", re.DOTALL)
_SMILES_RE = re.compile(r"<smiles>(.*?)</smiles>", re.DOTALL)
_SM_TOKEN_RE = re.compile(r"<sm_([^>]+)>")


# SECTION: Adapter


class UrsaAdapter:
    adapter_key = "ursa"

    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        target_id = source_key or "<unknown>"
        if not isinstance(raw_payload, list):
            raise adapter_schema_error(self.adapter_key, target_id, "expected a list of completion records")
        for row_index, raw_record in enumerate(raw_payload, start=1):
            if not isinstance(raw_record, dict):
                raise adapter_schema_error(
                    self.adapter_key,
                    target_id,
                    "each completion record must be a dict with a string 'completion' field",
                    record_index=row_index - 1,
                )
            raw_record = cast("dict[str, Any]", raw_record)
            if not isinstance(raw_record.get("completion"), str):
                raise adapter_schema_error(
                    self.adapter_key,
                    target_id,
                    "each completion record must be a dict with a string 'completion' field",
                    record_index=row_index - 1,
                )
            target_hint_smiles = _extract_target_hint(raw_record, self.adapter_key, target_id, row_index - 1)
            yield RawRouteEntry(
                payload=raw_record["completion"],
                source_key=source_key,
                source_row_index=row_index,
                target_hint_smiles=target_hint_smiles,
                source_order=row_index,
            )

    def cast(self, raw_route: Any, *, mode: AdaptMode = "strict", target: Target | None = None) -> Route:
        if not isinstance(raw_route, str):
            raise adapter_schema_error(
                self.adapter_key, target.id if target is not None else "<unknown>", "expected completion text"
            )
        if target is None:
            raise adapter_route_transform_error(self.adapter_key, "<unknown>", "ursa llm adaptation requires a target")

        expected_smiles = canonicalize_smiles(target.smiles)
        precursor_map = self._parse_completion(raw_route, mode=mode)
        if expected_smiles not in precursor_map:
            raise adapter_target_mismatch(
                self.adapter_key,
                target.id,
                expected_smiles=expected_smiles,
                actual_smiles=f"missing:{expected_smiles}",
            )
        route_target = build_molecule_from_precursor_map(
            expected_smiles, precursor_map, adapter=self.adapter_key, mode=mode
        )
        if route_target is None:
            raise AdapterLogicError(
                "URSA target molecule was pruned",
                code="adapter.target_pruned",
                context={"adapter": self.adapter_key, "target_id": target.id},
            )
        return Route(target=route_target)

    def _parse_completion(self, completion: str, *, mode: AdaptMode) -> dict[SmilesStr, list[str]]:
        cleaned = _THINK_BLOCK_RE.sub("", completion)
        precursor_map: dict[SmilesStr, list[str]] = {}
        for step_text in _SYNTHESIS_STEP_RE.findall(cleaned):
            product_match = _PRODUCT_RE.search(step_text)
            if product_match is None:
                continue
            product_raw = _extract_smiles(product_match.group(1))
            if product_raw is None:
                continue
            try:
                product_smiles = canonicalize_smiles(product_raw)
            except InvalidSmilesError:
                if mode == "strict":
                    raise
                continue
            reactants = [_extract_smiles(block) for block in _REACTANT_RE.findall(step_text)]
            valid_reactants = [reactant for reactant in reactants if reactant]
            if valid_reactants:
                precursor_map[product_smiles] = valid_reactants
        return precursor_map


# SECTION: Raw Helpers


def _extract_smiles(block: str) -> str | None:
    match = _SMILES_RE.search(block)
    if match is None:
        return None
    inner = match.group(1).strip()
    if "<sm_" in inner:
        tokens = _SM_TOKEN_RE.findall(inner)
        return "".join(tokens) if tokens else None
    return inner or None


def _extract_target_hint(
    raw_record: dict[str, Any],
    adapter_key: str,
    target_id: str,
    record_index: int,
) -> str | None:
    meta = raw_record.get("meta")
    if meta is None:
        return None
    if not isinstance(meta, dict):
        return None
    product_smiles = meta.get("product_smiles")
    if product_smiles is None:
        return None
    if not isinstance(product_smiles, str):
        raise adapter_schema_error(
            adapter_key,
            target_id,
            "completion record has non-string meta.product_smiles",
            record_index=record_index,
        )
    try:
        return canonicalize_smiles(product_smiles)
    except RetroCastException as exc:
        raise adapter_schema_error(
            adapter_key,
            target_id,
            "completion record has invalid meta.product_smiles",
            record_index=record_index,
        ) from exc
