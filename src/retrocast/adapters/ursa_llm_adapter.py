from __future__ import annotations

import gzip
import json
import logging
import re
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.adapters.errors import adapter_route_transform_error, adapter_schema_error, adapter_target_mismatch
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.io.blob import save_json_gz
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SYNTHESIS_STEP_RE = re.compile(r"<synthesis_step>(.*?)</synthesis_step>", re.DOTALL)
_PRODUCT_RE = re.compile(r"<product>(.*?)</product>", re.DOTALL)
_REACTANT_RE = re.compile(r"<reactant>(.*?)</reactant>", re.DOTALL)
_SMILES_RE = re.compile(r"<smiles>(.*?)</smiles>", re.DOTALL)
_SM_TOKEN_RE = re.compile(r"<sm_([^>]+)>")


class UrsaLlmCompletionInput(BaseModel):
    """Single prepared completion record consumed by the adapter."""

    model_config = ConfigDict(extra="ignore")

    completion: str


class UrsaLlmCompletionList(RootModel[list[UrsaLlmCompletionInput]]):
    """Prepared per-target completion payload consumed by the adapter."""


class UrsaLlmRawMeta(BaseModel):
    """Minimal raw metadata required to build the compatibility artifact."""

    model_config = ConfigDict(extra="ignore")

    product_smiles: str


class UrsaLlmRawRecord(BaseModel):
    """Single raw Ursa completion record."""

    model_config = ConfigDict(extra="ignore")

    meta: UrsaLlmRawMeta
    completion: str


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


def _detect_ursa_input_format(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if name.endswith(".json.gz"):
        return "json.gz"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    raise ValueError("unsupported input format; expected .json, .json.gz, .jsonl, or .jsonl.gz")


def _read_input_text(input_path: Path) -> str:
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return input_path.read_text(encoding="utf-8")


def load_ursa_llm_records(input_path: Path) -> list[Any]:
    """Load raw Ursa completion records from json/jsonl, optionally gzipped."""
    input_format = _detect_ursa_input_format(input_path)
    raw_text = _read_input_text(input_path).strip()
    if not raw_text:
        return []

    if input_format in {"json", "json.gz"}:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError("top-level JSON must be a list of records")
        return data

    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def prepare_ursa_llm_results(input_path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    """Build the current benchmark-centric compatibility artifact from raw Ursa completions."""
    records = load_ursa_llm_records(input_path)

    grouped_completions: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped_records = 0
    accepted_records = 0

    for raw_record in records:
        try:
            record = UrsaLlmRawRecord.model_validate(raw_record)
        except ValidationError:
            skipped_records += 1
            continue

        try:
            canonical_target = canonicalize_smiles(record.meta.product_smiles)
        except RetroCastException:
            skipped_records += 1
            continue

        grouped_completions[canonical_target].append({"completion": record.completion})
        accepted_records += 1

    results = {target_smiles: grouped_completions[target_smiles] for target_smiles in sorted(grouped_completions)}
    summary = {
        "solved_count": len(results),
        "total_records": len(records),
        "accepted_records": accepted_records,
        "skipped_records": skipped_records,
    }
    return results, summary


def write_prepared_ursa_llm_results(*, input_path: Path, output_dir: Path) -> dict[str, int]:
    """Persist the compatibility artifact and summary for raw Ursa completions."""
    results, summary = prepare_ursa_llm_results(input_path)
    results_path = output_dir / "results.json.gz"
    summary_path = output_dir / "summary.json"

    logger.info(
        "found %s unique targets across %s accepted completions.",
        summary["solved_count"],
        summary["accepted_records"],
    )
    if summary["skipped_records"]:
        logger.warning("skipped %s invalid or incomplete records.", summary["skipped_records"])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_gz(results, results_path)
    logger.info("successfully wrote pre-processed data to %s", results_path)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote summary to %s", summary_path)
    return summary


class UrsaLlmAdapter(BaseAdapter):
    """Adapter for Ursa LLM completions containing `<synthesis_step>` XML blocks."""

    adapter_key = "ursa-llm"

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        try:
            validated = UrsaLlmCompletionList.model_validate(raw_target_data)
        except ValidationError as e:
            raise adapter_schema_error(self.adapter_key, target.id, "invalid completion list") from e

        for rank, record in enumerate(validated.root, start=1):
            try:
                yield self._transform(record, target, rank=rank, ignore_stereo=ignore_stereo)
            except RetroCastException as e:
                logger.warning(f"  - completion #{rank} for '{target.id}' failed transformation: {e} [{e.code}]")
                continue

    def _transform(
        self, record: UrsaLlmCompletionInput, target: TargetIdentity, rank: int, ignore_stereo: bool = False
    ) -> Route:
        precursor_map = self._parse_completion(record.completion, ignore_stereo=ignore_stereo)
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
        return Route(target=molecule, rank=rank, metadata={})

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

__all__ = [
    "UrsaLlmAdapter",
    "prepare_ursa_llm_results",
    "write_prepared_ursa_llm_results",
    "load_ursa_llm_records",
]
