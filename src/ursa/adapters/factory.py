from ursa.adapters.aizynth_adapter import AizynthAdapter
from ursa.adapters.askcos_adapter import AskcosAdapter
from ursa.adapters.base_adapter import BaseAdapter
from ursa.adapters.dms_adapter import DMSAdapter
from ursa.adapters.dreamretro_adapter import DreamRetroAdapter
from ursa.adapters.multistepttl_adapter import TtlRetroAdapter
from ursa.adapters.retrochimera_adapter import RetrochimeraAdapter
from ursa.adapters.retrostar_adapter import RetroStarAdapter
from ursa.adapters.synplanner_adapter import SynPlannerAdapter
from ursa.adapters.syntheseus_adapter import SyntheseusAdapter
from ursa.exceptions import UrsaException

ADAPTER_MAP: dict[str, BaseAdapter] = {
    "aizynth": AizynthAdapter(),
    "askcos": AskcosAdapter(),
    "dms": DMSAdapter(),
    "dreamretro": DreamRetroAdapter(),
    "multistepttl": TtlRetroAdapter(),
    "retrochimera": RetrochimeraAdapter(),
    "retrostar": RetroStarAdapter(),
    "synplanner": SynPlannerAdapter(),
    "syntheseus": SyntheseusAdapter(),
}


def get_adapter(adapter_name: str) -> BaseAdapter:
    """
    retrieves an adapter instance based on its name from the config.
    """
    adapter = ADAPTER_MAP.get(adapter_name)
    if adapter is None:
        raise UrsaException(f"unknown adapter '{adapter_name}'. check `ursa-config.yaml` and `ADAPTER_MAP`.")
    return adapter
