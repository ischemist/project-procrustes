from __future__ import annotations

from typing import Any

from retrocast.exceptions import AdapterLogicError, AdapterSchemaError


def adapter_schema_error(adapter: str, target_id: str, detail: str, **context: Any) -> AdapterSchemaError:
    return AdapterSchemaError(
        f"raw data for target '{target_id}' failed {adapter} schema validation: {detail}",
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
        f"mismatched smiles for target {target_id}. expected canonical: {expected_smiles}, "
        f"but adapter produced: {actual_smiles}",
        code="adapter.target_mismatch",
        context={
            "adapter": adapter,
            "target_id": target_id,
            "expected_smiles": expected_smiles,
            "actual_smiles": actual_smiles,
        },
    )
