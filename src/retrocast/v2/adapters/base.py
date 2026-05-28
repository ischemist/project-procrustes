from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target

AdaptMode = Literal["strict", "prune"]


@dataclass(frozen=True, slots=True)
class RawRouteEntry:
    payload: Any
    source_key: str | None = None
    source_row_index: int | None = None
    source_record_id: str | None = None
    target_hint_id: str | None = None
    target_hint_smiles: str | None = None
    source_order: int | None = None


class Adapter(Protocol):
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]: ...

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route: ...
