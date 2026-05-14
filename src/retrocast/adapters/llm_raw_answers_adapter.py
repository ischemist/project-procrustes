from __future__ import annotations

import logging
import re
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from retrocast.adapters.base_adapter import BaseAdapter
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


class LlmRawAnswersInput(BaseModel):
    """Single LLM completion record. Extra fields are tolerated and ignored."""

    model_config = ConfigDict(extra="ignore")

    completion: str


class LlmRawAnswersList(RootModel[list[LlmRawAnswersInput]]):
    pass


def _extract_smiles(block: str) -> str | None:
    """Extract a SMILES from a `<product>`/`<reactant>` block.

    Supports raw SMILES inside `<smiles>...</smiles>` tags and the tokenized
    `<sm_X>` format produced by some LLMs.
    """
    match = _SMILES_RE.search(block)
    if not match:
        return None
    inner = match.group(1).strip()
    if "<sm_" in inner:
        tokens = _SM_TOKEN_RE.findall(inner)
        return "".join(tokens) if tokens else None
    return inner if inner else None


class LlmRawAnswersAdapter(BaseAdapter):
    """Adapter for LLM completions containing `<synthesis_step>` XML blocks.

    Expects a list of records `{"completion": "<answer>...<synthesis_step>..."}`,
    one per attempt for a single target. Yields one Route per successfully parsed
    completion, ranked by list order.
    """

    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        try:
            validated = LlmRawAnswersList.model_validate(raw_target_data)
        except ValidationError as e:
            raise adapter_schema_error("llm-raw-answers", target.id, "invalid completion list") from e

        for rank, record in enumerate(validated.root, start=1):
            try:
                yield self._transform(record, target, rank=rank, ignore_stereo=ignore_stereo)
            except RetroCastException as e:
                logger.warning(f"  - completion #{rank} for '{target.id}' failed transformation: {e} [{e.code}]")
                continue

    def _transform(
        self, record: LlmRawAnswersInput, target: TargetIdentity, rank: int, ignore_stereo: bool = False
    ) -> Route:
        precursor_map = self._parse_completion(record.completion, ignore_stereo=ignore_stereo)
        if not precursor_map:
            raise adapter_route_transform_error("llm-raw-answers", target.id, "no synthesis steps found in completion")

        expected_target = canonicalize_smiles(target.smiles, ignore_stereo=ignore_stereo)
        if expected_target not in precursor_map:
            raise adapter_target_mismatch(
                "llm-raw-answers",
                target.id,
                expected_smiles=expected_target,
                actual_smiles=f"missing:{expected_target}",
            )

        molecule = build_molecule_from_precursor_map(
            smiles=SmilesStr(expected_target), precursor_map=precursor_map, ignore_stereo=ignore_stereo
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
