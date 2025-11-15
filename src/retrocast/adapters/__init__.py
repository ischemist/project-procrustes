from typing import Any

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
from retrocast.schemas import Route, TargetInput

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


def adapt_single_route(
    raw_route: Any,
    target: TargetInput,
    adapter_name: str,
) -> Route | None:
    """
    Adapt a single raw route to the unified Route format.

    This is a convenience function for users who want to adapt individual routes
    programmatically without the full batch processing pipeline.

    Args:
        raw_route: A single route in the model's native format. For most adapters
            (route-centric like DMS, AiZynth), this is a single route object/dict.
            For target-centric adapters, this may be a dict containing the route data.
        target: Target molecule information (id and canonical SMILES).
        adapter_name: Name of the adapter to use (e.g., "dms", "aizynth", "retrostar").
            See ADAPTER_MAP.keys() for available adapters.

    Returns:
        Route object if successful, None if adaptation failed.

    Example:
        >>> from retrocast.adapters import adapt_single_route
        >>> from retrocast.schemas import TargetInput
        >>>
        >>> target = TargetInput(id="aspirin", smiles="CC(=O)Oc1ccccc1C(=O)O")
        >>> raw_dms_route = {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "children": [...]}
        >>>
        >>> route = adapt_single_route(raw_dms_route, target, "dms")
        >>> if route:
        ...     print(f"Route depth: {route.depth}")
        ...     print(f"Starting materials: {len(route.leaves)}")
    """
    adapter = get_adapter(adapter_name)

    # Wrap single route in a list since adapters expect list/dict format
    # Most adapters are route-centric and expect a list of routes
    raw_data = [raw_route] if not isinstance(raw_route, list) else raw_route

    # Get first successful route from the generator
    for route in adapter.adapt(raw_data, target):
        return route

    return None


def adapt_routes(
    raw_routes: Any,
    target: TargetInput,
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
        >>> from retrocast.schemas import TargetInput
        >>>
        >>> target = TargetInput(id="ibuprofen", smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        >>> raw_routes = [route1, route2, route3, ...]  # Your model's output
        >>>
        >>> routes = adapt_routes(raw_routes, target, "aizynth", max_routes=10)
        >>> print(f"Adapted {len(routes)} routes successfully")
    """
    adapter = get_adapter(adapter_name)
    routes = []

    for i, route in enumerate(adapter.adapt(raw_routes, target)):
        routes.append(route)
        if max_routes and i + 1 >= max_routes:
            break

    return routes


__all__ = [
    "adapt_single_route",
    "adapt_routes",
    "get_adapter",
    "ADAPTER_MAP",
    "BaseAdapter",
    "AizynthAdapter",
    "AskcosAdapter",
    "DMSAdapter",
    "DreamRetroAdapter",
    "TtlRetroAdapter",
    "PaRoutesAdapter",
    "RetrochimeraAdapter",
    "RetroStarAdapter",
    "SynPlannerAdapter",
    "SyntheseusAdapter",
    "SynLlaMaAdapter",
]
