from retrocast.adapters.base import AdaptMode
from retrocast.adapters.native import NativeAdapter, _call_native
from retrocast.typing import SmilesStr


class SynLlamaAdapter(NativeAdapter):
    adapter_slug = "synllama"

    def _parse_synthesis_string(
        self,
        synthesis_str: str,
        *,
        mode: AdaptMode = "strict",
    ) -> dict[SmilesStr, list[str]]:
        del self
        return _call_native("synllama_precursor_map_json", synthesis_str, mode=mode)
