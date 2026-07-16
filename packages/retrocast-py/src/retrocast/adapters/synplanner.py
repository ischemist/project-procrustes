import logging
from collections.abc import Iterator
from typing import Any

from retrocast.adapters.base import RawRouteEntry
from retrocast.adapters.native import NativeAdapter, _call_native, _json

logger = logging.getLogger(__name__)


class _RoutePayload(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class SynPlannerAdapter(NativeAdapter):
    adapter_slug = "synplanner"

    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        batch = _call_native("synplanner_entries_json", _json(raw_payload), source_key=source_key)
        target_id = source_key or "<unknown>"
        for skipped in batch["skipped"]:
            logger.warning(
                "skipping invalid synplanner route: target=%s source_index=%s source_order=%s error=%s",
                target_id,
                skipped["source_index"],
                skipped["source_order"],
                skipped["error"],
            )
        if batch["skipped"]:
            logger.warning(
                "skipped %s invalid synplanner route(s) for target %s; valid_routes=%s",
                len(batch["skipped"]),
                target_id,
                len(batch["entries"]),
            )
        for entry in batch["entries"]:
            entry["payload"] = _RoutePayload(entry["payload"])
            yield RawRouteEntry(**entry)
