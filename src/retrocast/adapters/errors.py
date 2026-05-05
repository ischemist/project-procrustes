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


def adapter_cycle_error(adapter: str, smiles: str) -> AdapterLogicError:
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} route contains a cycle involving SMILES: {smiles}",
        code="adapter.cycle_detected",
        context={"adapter": adapter, "smiles": smiles},
    )


def adapter_node_type_error(adapter: str, *, expected: str, actual: str, role: str) -> AdapterLogicError:
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} expected {role} node type '{expected}' but got '{actual}'",
        code="adapter.node_type_invalid",
        context={"adapter": adapter, "expected": expected, "actual": actual, "role": role},
    )


def adapter_missing_node_error(adapter: str, *, node_id: str, lookup: str, role: str) -> AdapterLogicError:
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} could not resolve {role} node '{node_id}' in {lookup}",
        code="adapter.node_missing",
        context={"adapter": adapter, "node_id": node_id, "lookup": lookup, "role": role},
    )


def adapter_route_string_error(
    adapter: str,
    detail: str,
    *,
    fragment: str | None = None,
    empty: bool = False,
) -> AdapterLogicError:
    context: dict[str, Any] = {"adapter": adapter}
    if fragment is not None:
        context["fragment"] = fragment
    code = "adapter.route_string_empty" if empty else "adapter.route_string_invalid"
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} route string is empty"
        if empty
        else f"{adapter_display_name(adapter)} route string is invalid: {detail}",
        code=code,
        context=context,
    )


def adapter_route_metadata_error(adapter: str, *, smiles: str, field: str) -> AdapterLogicError:
    return AdapterLogicError(
        f"{adapter_display_name(adapter)} non-leaf node for SMILES '{smiles}' is missing required route metadata field '{field}'",
        code="adapter.route_metadata_missing",
        context={"adapter": adapter, "smiles": smiles, "field": field},
    )
