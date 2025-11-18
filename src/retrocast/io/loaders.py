from pathlib import Path

from retrocast.io.files import load_json_gz
from retrocast.models.benchmark import BenchmarkSet
from retrocast.utils.logging import logger


def load_benchmark(path: Path) -> BenchmarkSet:
    """
    Loads a BenchmarkSet from a gzipped JSON file.
    """
    logger.info(f"Loading benchmark from {path}...")
    data = load_json_gz(path)
    benchmark = BenchmarkSet.model_validate(data)
    logger.info(f"Loaded benchmark '{benchmark.name}' with {len(benchmark.targets)} targets.")
    return benchmark


def load_raw_paroutes_list(path: Path) -> list[dict]:
    """
    Loads the raw PaRoutes list-of-dicts format.
    Used only during the initial curation phase.
    """
    logger.info(f"Loading raw PaRoutes data from {path}...")
    data = load_json_gz(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data
