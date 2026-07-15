from dataclasses import dataclass

from retrocast.adapters.base import AdaptMode
from retrocast.adapters.native import NativeAdapter, _call_native
from retrocast.typing import SmilesStr


@dataclass(slots=True)
class RetroStarParsedRoute:
    target_smiles: SmilesStr
    precursor_map: dict[SmilesStr, list[str]]
    step_scores: dict[SmilesStr, float]


class RetroStarAdapter(NativeAdapter):
    adapter_slug = "retrostar"

    def _parse_route_string(
        self,
        route_str: str,
        *,
        mode: AdaptMode = "strict",
    ) -> RetroStarParsedRoute:
        del self
        parsed = _call_native("reaction_string_parse_json", route_str, "retrostar", mode=mode)
        return RetroStarParsedRoute(**parsed)
