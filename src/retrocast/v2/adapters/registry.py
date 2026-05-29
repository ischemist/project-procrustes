from __future__ import annotations

from collections.abc import Callable

from retrocast.exceptions import AdapterResolutionError
from retrocast.v2.adapters.aizynth import AiZynthFinderAdapter
from retrocast.v2.adapters.askcos import AskcosAdapter
from retrocast.v2.adapters.base import Adapter
from retrocast.v2.adapters.dms import DirectMultiStepAdapter
from retrocast.v2.adapters.dreamretro import DreamRetroErAdapter
from retrocast.v2.adapters.molbuilder import MolBuilderAdapter
from retrocast.v2.adapters.multistepttl import MultiStepTTLAdapter
from retrocast.v2.adapters.paroutes import PaRoutesAdapter
from retrocast.v2.adapters.retrochimera import RetroChimeraAdapter
from retrocast.v2.adapters.retrostar import RetroStarAdapter
from retrocast.v2.adapters.synllama import SynLlamaAdapter
from retrocast.v2.adapters.synplanner import SynPlannerAdapter
from retrocast.v2.adapters.syntheseus import SyntheseusAdapter
from retrocast.v2.adapters.ursa import UrsaAdapter

AdapterFactory = Callable[[], Adapter]

ADAPTER_TYPES: dict[str, AdapterFactory] = {
    "aizynthfinder": AiZynthFinderAdapter,
    "askcos": AskcosAdapter,
    "directmultistep": DirectMultiStepAdapter,
    "dms": DirectMultiStepAdapter,
    "dreamretroer": DreamRetroErAdapter,
    "molbuilder": MolBuilderAdapter,
    "multistepttl": MultiStepTTLAdapter,
    "paroutes": PaRoutesAdapter,
    "retrochimera": RetroChimeraAdapter,
    "retrostar": RetroStarAdapter,
    "synllama": SynLlamaAdapter,
    "synplanner": SynPlannerAdapter,
    "syntheseus": SyntheseusAdapter,
    "ursa": UrsaAdapter,
}

DEPRECATED_ADAPTER_SLUGS = {
    "aizynth": "aizynthfinder",
    "dreamretro": "dreamretroer",
    "retro-star": "retrostar",
}


def normalize_adapter_slug(name: str) -> str:
    normalized = name.strip().lower()
    return DEPRECATED_ADAPTER_SLUGS.get(normalized, normalized)


def get_adapter(name: str) -> Adapter:
    slug = normalize_adapter_slug(name)
    try:
        return ADAPTER_TYPES[slug]()
    except KeyError as exc:
        raise AdapterResolutionError(
            f"Unknown v2 adapter: {name}",
            code="adapter.unknown",
            context={"adapter": name, "available_adapters": sorted(ADAPTER_TYPES)},
        ) from exc
