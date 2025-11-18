import gzip
from pathlib import Path

from pydantic import TypeAdapter

from retrocast.models.chem import Route
from retrocast.utils.logging import logger

# Pre-define the adapter for performance and reuse
RoutesDict = dict[str, list[Route]]
_ROUTES_ADAPTER = TypeAdapter(RoutesDict)


def save_routes(routes: RoutesDict, path: Path) -> None:
    """
    Saves a dictionary of routes to a gzipped JSON file.

    Args:
        routes: dict mapping target_id -> list[Route]
        path: output path (usually .json.gz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # dump_json returns bytes, so we use "wb"
        json_bytes = _ROUTES_ADAPTER.dump_json(routes, indent=2)
        with gzip.open(path, "wb") as f:
            f.write(json_bytes)
        logger.debug(f"Saved {sum(len(r) for r in routes.values())} routes to {path}")
    except Exception as e:
        logger.error(f"Failed to save routes to {path}: {e}")
        raise


def load_routes(path: Path) -> RoutesDict:
    """
    Loads routes from a gzipped JSON file.

    Returns:
        dict mapping target_id -> list[Route]
    """
    path = Path(path)
    logger.debug(f"Loading routes from {path}...")

    try:
        with gzip.open(path, "rb") as f:
            json_bytes = f.read()

        routes = _ROUTES_ADAPTER.validate_json(json_bytes)
        logger.debug(f"Loaded {sum(len(r) for r in routes.values())} routes for {len(routes)} targets.")
        return routes
    except Exception as e:
        logger.error(f"Failed to load routes from {path}: {e}")
        raise
