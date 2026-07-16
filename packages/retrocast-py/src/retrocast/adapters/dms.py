import json

from pydantic import BaseModel, Field

from retrocast.adapters.native import NativeAdapter


class DMSTree(BaseModel):
    smiles: str
    children: list["DMSTree"] = Field(default_factory=list)


class DirectMultiStepAdapter(NativeAdapter):
    adapter_slug = "directmultistep"

    @staticmethod
    def calculate_route_length(dms_node: DMSTree) -> int:
        from retrocast import _native

        return _native.dms_route_length_json(json.dumps(dms_node.model_dump(mode="json"), separators=(",", ":")))
