from __future__ import annotations

from typing import Any

from retrocast.exceptions import AdapterLogicError, AdapterSchemaError

ADAPTER_DISPLAY_NAMES = {
    "aizynth": "AiZynthFinder",
    "askcos": "ASKCOS",
    "dms": "DMS",
    "dreamretro": "DreamRetro",
    "molbuilder": "MolBuilder",
    "multistepttl": "MultiStepTTL",
    "paroutes": "PaRoutes",
    "retrochimera": "RetroChimera",
    "retrostar": "RetroStar",
    "synllama": "SynLlama",
    "synplanner": "SynPlanner",
    "syntheseus": "Syntheseus",
}


def adapter_display_name(adapter: str) -> str:
    return ADAPTER_DISPLAY_NAMES.get(adapter, adapter)


def adapter_schema_error(adapter: str, target_id: str, detail: str, **context: Any) -> AdapterSchemaError:
    return AdapterSchemaError(
        f"raw data for target '{target_id}' failed {adapter_display_name(adapter)} schema validation: {detail}",
        code="adapter.schema_invalid",
        context={"adapter": adapter, "target_id": target_id, **context},
    )


def adapter_target_mismatch(
    adapter: str,
    target_id: str,
    *,
    expected_smiles: str,
    actual_smiles: str,
) -> AdapterLogicError:
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} produced mismatched SMILES for target {target_id}. "
        f"expected canonical: {expected_smiles}, but adapter produced: {actual_smiles}",
        code="adapter.target_mismatch",
        context={
            "adapter": adapter,
            "target_id": target_id,
            "expected_smiles": expected_smiles,
            "actual_smiles": actual_smiles,
        },
    )
