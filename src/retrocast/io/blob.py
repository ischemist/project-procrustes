import gzip
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def save_json_gz(data: Any, path: Path) -> None:
    """
    Serializes data to a gzipped JSON file.
    If data is a Pydantic model, dumps it first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        json_obj = data.model_dump(mode="json")
    else:
        json_obj = data

    try:
        json_str = json.dumps(json_obj, indent=2)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_str)
        logger.debug(f"Saved {path}")
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        raise


def load_json_gz(path: Path) -> Any:
    """Loads a gzipped JSON file."""
    path = Path(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise
