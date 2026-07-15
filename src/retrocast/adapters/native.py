from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, ClassVar

from pydantic import BaseModel

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.exceptions import (
    AdapterLogicError,
    AdapterSchemaError,
    InvalidSmilesError,
    RetroCastException,
    UnsupportedAdapterFeatureError,
)
from retrocast.models.route import Route
from retrocast.models.task import Target


class NativeAdapter:
    adapter_slug: ClassVar[str]

    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        result = _call_native(
            "adapter_entries_json",
            _json(raw_payload),
            self.adapter_slug,
            source_key=source_key,
        )
        for entry in result:
            yield RawRouteEntry(**entry)

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route:
        result = _call_native(
            "adapter_cast_result_json",
            _json(raw_route),
            self.adapter_slug,
            mode=mode,
            target_json=target.model_dump_json(exclude_none=True) if target is not None else None,
        )
        return Route.model_validate(result)


def _call_native(function_name: str, *args: Any, **kwargs: Any) -> Any:
    from retrocast import _native

    payload = json.loads(getattr(_native, function_name)(*args, **kwargs))
    failure = payload["failure"]
    if failure is not None:
        raise _boundary_error(failure)
    return payload["value"]


def _boundary_error(failure: dict[str, Any]) -> RetroCastException:
    code = failure["code"]
    message = failure.get("message") or code
    context = failure.get("context") or {}
    if code == "adapter.schema_invalid":
        return AdapterSchemaError(message, code=code, context=context)
    if code == "chem.invalid_smiles":
        return InvalidSmilesError(message, code=code, context=context)
    if code == "adapter.unsupported_feature":
        return UnsupportedAdapterFeatureError(message, code=code, context=context)
    return AdapterLogicError(message, code=code, context=context)


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    to_json = getattr(value, "_retrocast_json", None)
    if to_json is not None:
        return to_json()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
