from typing import Any

from retrocast._warnings import warn_deprecated
from retrocast.adapters.aizynth_adapter import AiZynthFinderAdapter
from retrocast.adapters.askcos_adapter import AskcosAdapter
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.dms_adapter import DirectMultiStepAdapter
from retrocast.adapters.dreamretro_adapter import DreamRetroErAdapter
from retrocast.adapters.molbuilder_adapter import MolBuilderAdapter
from retrocast.adapters.multistepttl_adapter import MultiStepTTLAdapter
from retrocast.adapters.paroutes_adapter import PaRoutesAdapter
from retrocast.adapters.retrochimera_adapter import RetroChimeraAdapter
from retrocast.adapters.retrostar_adapter import RetroStarAdapter
from retrocast.adapters.synllama_adapter import SynLlamaAdapter
from retrocast.adapters.synplanner_adapter import SynPlannerAdapter
from retrocast.adapters.syntheseus_adapter import SyntheseusAdapter
from retrocast.adapters.ursa_llm_adapter import UrsaAdapter
from retrocast.exceptions import AdapterError, AdapterResolutionError, ChemError
from retrocast.models.chem import Route, TargetIdentity
from retrocast.workflow.adapt import adapt_target_routes

ADAPTER_TYPES: dict[str, type[BaseAdapter]] = {
    "aizynthfinder": AiZynthFinderAdapter,
    "askcos": AskcosAdapter,
    "directmultistep": DirectMultiStepAdapter,
    "dreamretroer": DreamRetroErAdapter,
    "molbuilder": MolBuilderAdapter,
    "multistepttl": MultiStepTTLAdapter,
    "paroutes": PaRoutesAdapter,
    "retrochimera": RetroChimeraAdapter,
    "retrostar": RetroStarAdapter,
    "synplanner": SynPlannerAdapter,
    "syntheseus": SyntheseusAdapter,
    "synllama": SynLlamaAdapter,
    "ursa": UrsaAdapter,
}

ADAPTER_MAP: dict[str, BaseAdapter] = {
    adapter_name: adapter_type() for adapter_name, adapter_type in ADAPTER_TYPES.items()
}

DEPRECATED_ADAPTER_SLUGS: dict[str, str] = {
    "aizynth": "aizynthfinder",
    "dms": "directmultistep",
    "dreamretro": "dreamretroer",
    "ursa-llm": "ursa",
}

_DEPRECATED_ADAPTER_ALIASES: dict[str, type[BaseAdapter]] = {
    "AizynthAdapter": AiZynthFinderAdapter,
    "DMSAdapter": DirectMultiStepAdapter,
    "DreamRetroAdapter": DreamRetroErAdapter,
    "TtlRetroAdapter": MultiStepTTLAdapter,
    "RetrochimeraAdapter": RetroChimeraAdapter,
    "SynLLaMaAdapter": SynLlamaAdapter,
    "SynLlaMaAdapter": SynLlamaAdapter,
    "UrsaLlmAdapter": UrsaAdapter,
}

_DEPRECATED_ADAPTER_REPLACEMENTS = {
    "AizynthAdapter": "AiZynthFinderAdapter",
    "DMSAdapter": "DirectMultiStepAdapter",
    "DreamRetroAdapter": "DreamRetroErAdapter",
    "TtlRetroAdapter": "MultiStepTTLAdapter",
    "RetrochimeraAdapter": "RetroChimeraAdapter",
    "SynLLaMaAdapter": "SynLlamaAdapter",
    "SynLlaMaAdapter": "SynLlamaAdapter",
    "UrsaLlmAdapter": "UrsaAdapter",
}


def __getattr__(name: str) -> Any:
    adapter_type = _DEPRECATED_ADAPTER_ALIASES.get(name)
    if adapter_type is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        old=f"{__name__}.{name}",
        new=f"{__name__}.{_DEPRECATED_ADAPTER_REPLACEMENTS[name]}",
        remove_in="0.7",
        stacklevel=2,
    )
    globals()[name] = adapter_type
    return adapter_type


def get_adapter(adapter_name: str) -> BaseAdapter:
    """
    Retrieves a fresh adapter instance from the adapter registry.
    """
    canonical_name = normalize_adapter_slug(adapter_name)
    adapter_type = ADAPTER_TYPES.get(canonical_name)
    if adapter_type is None:
        raise AdapterResolutionError(
            f"unknown adapter '{adapter_name}'. Check `retrocast.adapters.ADAPTER_MAP` for available adapters.",
            code="adapter.unknown",
            context={
                "adapter": adapter_name,
                "available": sorted(ADAPTER_TYPES.keys()),
                "deprecated_aliases": sorted(DEPRECATED_ADAPTER_SLUGS.keys()),
            },
        )
    return adapter_type()


def normalize_adapter_slug(adapter_name: str) -> str:
    """Return the canonical adapter slug, warning for deprecated pre-0.7 slugs."""
    canonical_name = DEPRECATED_ADAPTER_SLUGS.get(adapter_name)
    if canonical_name is None:
        return adapter_name
    warn_deprecated(
        old=f"adapter slug '{adapter_name}'",
        new=f"'{canonical_name}'",
        remove_in="0.7",
        note="Update --adapter flags and manifest directives.",
        stacklevel=3,
    )
    return canonical_name


def all_adapter_slugs() -> list[str]:
    """Return canonical and deprecated adapter slugs accepted by get_adapter."""
    return sorted([*ADAPTER_TYPES.keys(), *DEPRECATED_ADAPTER_SLUGS.keys()])


def adapt_single_route(
    raw_route: Any,
    target: TargetIdentity,
    adapter_name: str,
) -> Route | None:
    """Deprecated wrapper returning the first route from a target-local payload."""
    warn_deprecated(
        old="adapt_single_route",
        new="adapt_route(...)",
        remove_in="0.7",
        note="This wrapper returns the first successful route or None; "
        "use target-free `adapt_route(...)` for single-route adaptation.",
        stacklevel=2,
    )
    adapter = get_adapter(adapter_name)
    try:
        prediction = next(adapt_target_routes(adapter, raw_route, target), None)
        return prediction.route if prediction is not None else None
    except (AdapterError, ChemError):
        return None


def adapt_routes(
    raw_routes: Any,
    target: TargetIdentity,
    adapter_name: str,
    max_routes: int | None = None,
) -> list[Route]:
    """
    Deprecated target-local wrapper for adapting multiple raw routes.

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
        >>> routes = adapt_routes(raw_routes, target, "aizynthfinder", max_routes=10)
        >>> print(f"Adapted {len(routes)} routes successfully")
    """
    warn_deprecated(
        old="adapt_routes",
        new="adapt_provider_output(...) or adapt_target_keyed_provider_output(...)",
        remove_in="0.7",
        note="Target-local adaptation is being removed from the public API. "
        "Use provider-output workflows for route standardization.",
        stacklevel=2,
    )
    adapter = get_adapter(adapter_name)
    routes = []
    for i, prediction in enumerate(adapt_target_routes(adapter, raw_routes, target)):
        routes.append(prediction.route)
        if max_routes and i + 1 >= max_routes:
            break

    return routes


__all__ = [
    "adapt_single_route",
    "adapt_routes",
    "get_adapter",
    "normalize_adapter_slug",
    "all_adapter_slugs",
    "ADAPTER_MAP",
    "ADAPTER_TYPES",
    "DEPRECATED_ADAPTER_SLUGS",
    "BaseAdapter",
    "AiZynthFinderAdapter",
    "AizynthAdapter",
    "AskcosAdapter",
    "DirectMultiStepAdapter",
    "DMSAdapter",
    "DreamRetroErAdapter",
    "DreamRetroAdapter",
    "UrsaAdapter",
    "UrsaLlmAdapter",
    "MolBuilderAdapter",
    "MultiStepTTLAdapter",
    "TtlRetroAdapter",
    "PaRoutesAdapter",
    "RetroChimeraAdapter",
    "RetrochimeraAdapter",
    "RetroStarAdapter",
    "SynPlannerAdapter",
    "SyntheseusAdapter",
    "SynLlamaAdapter",
    "SynLLaMaAdapter",
    "SynLlaMaAdapter",
]
