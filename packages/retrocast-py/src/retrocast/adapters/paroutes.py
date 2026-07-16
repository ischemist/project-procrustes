import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, Field

from retrocast.adapters.base import RawRouteEntry
from retrocast.adapters.native import NativeAdapter
from retrocast.exceptions import AdapterSchemaError


class ConditionSlotParseStatistics(BaseModel):
    malformed_rsmi_count: int = 0
    uncanonicalizable_token_count: int = 0
    uncanonicalizable_tokens: defaultdict[str, int] = Field(default_factory=lambda: defaultdict(int))

    @property
    def distinct_uncanonicalizable_token_count(self) -> int:
        return len(self.uncanonicalizable_tokens)

    @property
    def top_uncanonicalizable_tokens(self) -> list[tuple[str, int]]:
        return sorted(self.uncanonicalizable_tokens.items(), key=lambda item: (-item[1], item[0]))[:5]


class PaRoutesAdapter(NativeAdapter):
    adapter_slug = "paroutes"

    def iter_raw_routes(self, raw_payload: Any, *, source_key: str | None = None) -> Iterator[RawRouteEntry]:
        if isinstance(raw_payload, Mapping):
            looks_like_route = raw_payload.get("type") == "mol" or any(
                key in raw_payload for key in ("smiles", "children")
            )
            if not looks_like_route and any(not isinstance(key, str) for key in raw_payload):
                raise AdapterSchemaError(
                    "PaRoutes target route mapping keys must be strings",
                    code="adapter.schema_invalid",
                    context={"adapter": self.adapter_slug, "source_key": source_key},
                )
        yield from super().iter_raw_routes(raw_payload, source_key=source_key)


def analyze_condition_slots(raw_route: dict[str, Any], *, stats: ConditionSlotParseStatistics) -> None:
    from retrocast import _native

    payload = _native.paroutes_condition_stats_json(
        json.dumps(raw_route, separators=(",", ":")),
        stats.model_dump_json(),
    )
    updated = ConditionSlotParseStatistics.model_validate_json(payload)
    stats.malformed_rsmi_count = updated.malformed_rsmi_count
    stats.uncanonicalizable_token_count = updated.uncanonicalizable_token_count
    stats.uncanonicalizable_tokens = updated.uncanonicalizable_tokens
