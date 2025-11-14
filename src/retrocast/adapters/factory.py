from retrocast.adapters.aizynth_adapter import AizynthAdapter
from retrocast.adapters.askcos_adapter import AskcosAdapter
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.dms_adapter import DMSAdapter
from retrocast.adapters.dreamretro_adapter import DreamRetroAdapter
from retrocast.adapters.multistepttl_adapter import TtlRetroAdapter
from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.adapters.retrochimera_adapter import RetrochimeraAdapter
from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.adapters.synllama_adapter import SynLlaMaAdapter
from retrocast.adapters.synplanner_adapter import SynPlannerAdapter
from retrocast.adapters.syntheseus_adapter import SyntheseusAdapter
from retrocast.exceptions import RetroCastException

ADAPTER_MAP: dict[str, BaseAdapter] = {
    "aizynth": AizynthAdapter(),
    "askcos": AskcosAdapter(),
    "dms": DMSAdapter(),
    "dreamretro": DreamRetroAdapter(),
    "multistepttl": TtlRetroAdapter(),
    "paroutes": PaRoutesAdapter(),
    "retrochimera": RetrochimeraAdapter(),
    "retrostar": RetroStarAdapter(),
    "synplanner": SynPlannerAdapter(),
    "syntheseus": SyntheseusAdapter(),
    "synllama": SynLlaMaAdapter(),
}


def get_adapter(adapter_name: str) -> BaseAdapter:
    """
    retrieves an adapter instance based on its name from the config.
    """
    adapter = ADAPTER_MAP.get(adapter_name)
    if adapter is None:
        raise RetroCastException(f"unknown adapter '{adapter_name}'. check `retrocast-config.yaml` and `ADAPTER_MAP`.")
    return adapter
