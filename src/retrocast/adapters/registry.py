from __future__ import annotations

from collections.abc import Callable

from retrocast.adapters.aizynth import AiZynthFinderAdapter
from retrocast.adapters.askcos import AskcosAdapter
from retrocast.adapters.base import Adapter
from retrocast.adapters.dms import DirectMultiStepAdapter
from retrocast.adapters.dreamretro import DreamRetroErAdapter
from retrocast.adapters.molbuilder import MolBuilderAdapter
from retrocast.adapters.multistepttl import MultiStepTTLAdapter
from retrocast.adapters.paroutes import PaRoutesAdapter
from retrocast.adapters.retrochimera import RetroChimeraAdapter
from retrocast.adapters.retrostar import RetroStarAdapter
from retrocast.adapters.synllama import SynLlamaAdapter
from retrocast.adapters.synplanner import SynPlannerAdapter
from retrocast.adapters.syntheseus import SyntheseusAdapter
from retrocast.adapters.ursa import UrsaAdapter
from retrocast.exceptions import AdapterResolutionError

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
