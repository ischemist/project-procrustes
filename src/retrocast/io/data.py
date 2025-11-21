import gzip
from pathlib import Path

from pydantic import TypeAdapter

from retrocast.exceptions import RetroCastIOError
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.models.chem import Route
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelStatistics
from retrocast.typing import SmilesStr
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


def load_stock_file(path: Path) -> set[SmilesStr]:
    """
    Loads a set of stock SMILES from a text file (one per line).
    Assumes the file is already canonicalized.
    """
    logger.debug(f"Loading stock from {path}...")
    try:
        with path.open("r", encoding="utf-8") as f:
            stock = {line.strip() for line in f if line.strip()}

        logger.info(f"Loaded {len(stock):,} molecules from stock file.")
        return stock

    except OSError as e:
        logger.error(f"Failed to read stock file: {path}")
        raise RetroCastIOError(f"Stock loading error on {path}: {e}") from e


def load_execution_stats(path: Path) -> ExecutionStats:
    """
    Loads ExecutionStats from a gzipped JSON file.
    """
    logger.info(f"Loading execution stats from {path}...")
    data = load_json_gz(path)
    stats = ExecutionStats.model_validate(data)
    logger.info(
        f"Loaded execution stats with {len(stats.wall_time)} wall_time and {len(stats.cpu_time)} cpu_time entries."
    )
    return stats


def save_execution_stats(stats: ExecutionStats, path: Path) -> None:
    """
    Saves ExecutionStats to a gzipped JSON file.
    """
    logger.info(f"Saving execution stats to {path}...")
    save_json_gz(stats, path)
    logger.info(
        f"Saved execution stats with {len(stats.wall_time)} wall_time and {len(stats.cpu_time)} cpu_time entries."
    )


class BenchmarkResultsLoader:
    """
    Access point for loaded benchmark data.

    Directory Structure Assumption:
      data/
        4-scored/  {benchmark}/{model}/{stock}/evaluation.json.gz
        5-results/ {benchmark}/{model}/{stock}/statistics.json.gz
    """

    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.results_dir = self.root / "5-results"
        self.scored_dir = self.root / "4-scored"

    def load_statistics(self, benchmark: str, models: list[str], stock: str = "n5-stock") -> list[ModelStatistics]:
        """
        Loads pre-computed statistics for a list of models.
        Returns only successfully loaded objects.
        """
        loaded = []
        for model in models:
            path = self.results_dir / benchmark / model / stock / "statistics.json.gz"

            if not path.exists():
                logger.warning(f"[yellow]Missing statistics[/]: {model} ({path.name})")
                continue

            try:
                raw = load_json_gz(path)
                stats = ModelStatistics.model_validate(raw)
                loaded.append(stats)
            except Exception as e:
                logger.error(f"[red]Failed to load {model}[/]: {e}")

        return loaded

    def load_evaluation(self, benchmark: str, model: str, stock: str = "n5-stock") -> EvaluationResults | None:
        """
        Loads raw scored evaluation results for a single model.
        """
        path = self.scored_dir / benchmark / model / stock / "evaluation.json.gz"

        if not path.exists():
            logger.warning(f"[yellow]Missing evaluation[/]: {model}")
            return None

        try:
            raw = load_json_gz(path)
            return EvaluationResults.model_validate(raw)
        except Exception as e:
            logger.error(f"[red]Failed to load {model}[/]: {e}")
            return None
