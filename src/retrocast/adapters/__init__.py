from typing import Any

from retrocast.adapters.aizynth_adapter import AizynthAdapter
from retrocast.adapters.askcos_adapter import AskcosAdapter
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.dms_adapter import DMSAdapter
from retrocast.adapters.dreamretro_adapter import DreamRetroAdapter
from retrocast.adapters.molbuilder_adapter import MolBuilderAdapter
from retrocast.adapters.multistepttl_adapter import TtlRetroAdapter
from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.adapters.retrochimera_adapter import RetrochimeraAdapter
from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.adapters.synllama_adapter import SynLlaMaAdapter
from retrocast.adapters.synplanner_adapter import SynPlannerAdapter
from retrocast.adapters.syntheseus_adapter import SyntheseusAdapter
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import Route, TargetIdentity

ADAPTER_MAP: dict[str, BaseAdapter] = {
    "aizynth": AizynthAdapter(),
    "askcos": AskcosAdapter(),
    "dms": DMSAdapter(),
    "dreamretro": DreamRetroAdapter(),
    "molbuilder": MolBuilderAdapter(),
    "multistepttl": TtlRetroAdapter(),
    "paroutes": PaRoutesAdapter(),
    "retrochimera": RetrochimeraAdapter(),
    "retrostar": RetroStarAdapter(),
    "synplanner": SynPlannerAdapter(),
    "syntheseus": SyntheseusAdapter(),
    "synllama": SynLlaMaAdapter(),
}

# Adapters that expect target-centric data format (dict with metadata + nested routes)
# vs route-centric format (list of route objects)
TARGET_CENTRIC_ADAPTERS = {"askcos", "retrochimera", "paroutes"}


def get_adapter(adapter_name: str) -> BaseAdapter:
    """
    Retrieves an adapter instance from the `ADAPTER_MAP`.
    """
    adapter = ADAPTER_MAP.get(adapter_name)
    if adapter is None:
        raise RetroCastException(
            f"unknown adapter '{adapter_name}'. Check `retrocast.adapters.ADAPTER_MAP` for available adapters."
        )
    return adapter


def adapt_single_route(
    raw_route: Any,
    target: TargetIdentity,
    adapter_name: str,
) -> Route | None:
    """Adapt a single raw route to the unified Route format."""
    adapter = get_adapter(adapter_name)

    if adapter_name in TARGET_CENTRIC_ADAPTERS:
        raw_data = raw_route
    else:
        raw_data = [raw_route] if not isinstance(raw_route, list) else raw_route

    return next(adapter.cast(raw_data, target), None)


def adapt_routes(
    raw_routes: Any,
    target: TargetIdentity,
    adapter_name: str,
    max_routes: int | None = None,
) -> list[Route]:
    """
    Adapt multiple raw routes to the unified Route format.

    Args:
        raw_routes: Routes in the model's native format (typically a list or dict).
        target: Target molecule information (id and canonical SMILES).
        adapter_name: Name of the adapter to use.
        max_routes: Maximum number of routes to return (None for all successful routes).

    Returns:
        List of successfully adapted Route objects.

    Example:
        >>> from retrocast.adapters import adapt_routes
        >>> from retrocast.models.chem import TargetIdentity
        >>>
        >>> target = TargetIdentity(id="ibuprofen", smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        >>> raw_routes = [route1, route2, route3, ...]  # Your model's output
        >>>
        >>> routes = adapt_routes(raw_routes, target, "aizynth", max_routes=10)
        >>> print(f"Adapted {len(routes)} routes successfully")
    """
    adapter = get_adapter(adapter_name)
    routes = []

    for i, route in enumerate(adapter.cast(raw_routes, target)):
        routes.append(route)
        if max_routes and i + 1 >= max_routes:
            break

    return routes


__all__ = [
    "adapt_single_route",
    "adapt_routes",
    "get_adapter",
    "ADAPTER_MAP",
    "TARGET_CENTRIC_ADAPTERS",
    "BaseAdapter",
    "AizynthAdapter",
    "AskcosAdapter",
    "DMSAdapter",
    "DreamRetroAdapter",
    "MolBuilderAdapter",
    "TtlRetroAdapter",
    "PaRoutesAdapter",
    "RetrochimeraAdapter",
    "RetroStarAdapter",
    "SynPlannerAdapter",
    "SyntheseusAdapter",
    "SynLlaMaAdapter",
]
