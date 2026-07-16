from dataclasses import dataclass

from retrocast.adapters.base import AdaptMode
from retrocast.adapters.native import NativeAdapter, _call_native
from retrocast.typing import SmilesStr


@dataclass(slots=True)
class DreamRetroParsedRoute:
    target_smiles: SmilesStr
    precursor_map: dict[SmilesStr, list[str]]


class DreamRetroErAdapter(NativeAdapter):
    adapter_slug = "dreamretroer"

    def _parse_route_string(
        self,
        route_str: str,
        *,
        mode: AdaptMode = "strict",
    ) -> DreamRetroParsedRoute:
        del self
        parsed = _call_native("reaction_string_parse_json", route_str, "dreamretroer", mode=mode)
        return DreamRetroParsedRoute(
            target_smiles=parsed["target_smiles"],
            precursor_map=parsed["precursor_map"],
        )
