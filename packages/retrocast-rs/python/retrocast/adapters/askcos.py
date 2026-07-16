from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from retrocast.adapters.base import RawRouteEntry
from retrocast.adapters.native import NativeAdapter, _call_native, _json

ASKCOS_ROOT_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True, slots=True)
class AskcosPathwayEdge:
    source: str
    target: str


@dataclass(frozen=True, slots=True, eq=False)
class AskcosPathwayPayload:
    pathway_edges: tuple[AskcosPathwayEdge, ...]
    uuid2smiles: Mapping[str, str]
    node_dict: Mapping[str, Any]
    annotations: Mapping[str, Any]

    def _retrocast_json(self) -> dict[str, Any]:
        return {
            "pathway_edges": [{"source": edge.source, "target": edge.target} for edge in self.pathway_edges],
            "uuid2smiles": dict(self.uuid2smiles),
            "node_dict": dict(self.node_dict),
            "annotations": dict(self.annotations),
        }


class AskcosAdapter(NativeAdapter):
    adapter_slug = "askcos"

    def __init__(self, use_full_graph: bool = False) -> None:
        self.use_full_graph = use_full_graph

    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        adapter = "askcosfullgraph" if self.use_full_graph else self.adapter_slug
        entries = _call_native(
            "adapter_entries_json",
            _json(raw_payload),
            adapter,
            source_key=source_key,
        )
        for entry in entries:
            payload = entry["payload"]
            entry["payload"] = AskcosPathwayPayload(
                pathway_edges=tuple(AskcosPathwayEdge(**edge) for edge in payload["pathway_edges"]),
                uuid2smiles=MappingProxyType(payload["uuid2smiles"]),
                node_dict=MappingProxyType(payload["node_dict"]),
                annotations=MappingProxyType(payload["annotations"]),
            )
            yield RawRouteEntry(**entry)
