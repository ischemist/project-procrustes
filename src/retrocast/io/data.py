import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Literal, overload

from pydantic import TypeAdapter

from retrocast.exceptions import RetroCastIOError
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.provenance import create_manifest
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.models.chem import Route, StockStatistics
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.stats import ModelStatistics
from retrocast.typing import InchiKeyStr, SmilesStr

logger = logging.getLogger(__name__)

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

    # dump_json returns bytes, so we use "wb"
    json_bytes = _ROUTES_ADAPTER.dump_json(routes, indent=2)
    with gzip.open(path, "wb") as f:
        f.write(json_bytes)
    logger.debug(f"Saved {sum(len(r) for r in routes.values())} routes to {path}")


def load_routes(path: Path) -> RoutesDict:
    """
    Loads routes from a gzipped JSON file.

    Returns:
        dict mapping target_id -> list[Route]
    """
    path = Path(path)
    logger.debug(f"Loading routes from {path}...")

    with gzip.open(path, "rb") as f:
        json_bytes = f.read()

    routes = _ROUTES_ADAPTER.validate_json(json_bytes)
    logger.debug(f"Loaded {sum(len(r) for r in routes.values())} routes for {len(routes)} targets.")
    return routes


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


@overload
def load_stock_file(path: Path, return_as: Literal["inchikey"] = "inchikey") -> set[InchiKeyStr]: ...
@overload
def load_stock_file(path: Path, return_as: Literal["smiles"]) -> set[SmilesStr]: ...
def load_stock_file(
    path: Path, return_as: Literal["inchikey", "smiles"] = "inchikey"
) -> set[InchiKeyStr] | set[SmilesStr]:
    """
    Loads a set of stock molecules from a CSV.GZ file.

    Expects gzipped CSV format with header: SMILES,InChIKey

    Args:
        path: Path to stock file (.csv.gz)
        return_as: Format to return stock in - "inchikey" (default) or "smiles"

    Returns:
        Set of InChI keys or SMILES representing available stock molecules

    Raises:
        RetroCastIOError: If file cannot be read, format is invalid, or return_as is invalid
    """
    path = Path(path)

    if return_as not in ("inchikey", "smiles"):
        raise RetroCastIOError(f"Invalid return_as parameter: '{return_as}'. Must be 'inchikey' or 'smiles'.")

    if not path.exists():
        raise RetroCastIOError(
            f"Stock file not found: {path}. "
            f"Expected .csv.gz format. Run scripts/1-canonicalize-stock.py to generate stock files."
        )

    if not path.name.endswith(".csv.gz"):
        raise RetroCastIOError(
            f"Invalid stock file format: {path}. "
            f"Only .csv.gz format is supported. Run scripts/1-canonicalize-stock.py to convert."
        )

    logger.debug(f"Loading stock from {path}...")

    try:
        molecules = set()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate header
            required_col = "InChIKey" if return_as == "inchikey" else "SMILES"
            if reader.fieldnames is None or required_col not in reader.fieldnames:
                raise RetroCastIOError(
                    f"Invalid stock CSV format. Expected header with '{required_col}' column. Got: {reader.fieldnames}"
                )

            for row in reader:
                if return_as == "inchikey":
                    value = row.get("InChIKey", "").strip()
                    if value:
                        molecules.add(InchiKeyStr(value))
                elif return_as == "smiles":
                    value = row.get("SMILES", "").strip()
                    if value:
                        molecules.add(SmilesStr(value))

        logger.info(f"Loaded {len(molecules):,} molecules from {path.name}")
        return molecules

    except OSError as e:
        raise RetroCastIOError(f"Failed to read stock file {path}: {e}") from e


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


def save_stock_files(
    stock: dict[InchiKeyStr, SmilesStr],
    stock_name: str,
    output_dir: Path,
    source_path: Path | None = None,
    statistics: StockStatistics | None = None,
) -> tuple[Path, Path, Path]:
    """
    Saves stock in dual format: CSV (with InChI keys) and TXT (SMILES only).

    Also generates a manifest file with provenance tracking, file hashes, and statistics.

    Args:
        stock: Dictionary mapping InChIKey -> canonical SMILES
        stock_name: Base name for output files (without extension)
        output_dir: Directory to write output files
        source_path: Optional path to source file for provenance tracking
        statistics: Optional StockStatistics for manifest

    Returns:
        Tuple of (csv_path, txt_path, manifest_path)

    Example:
        stock = {
            "UHOVQNZJYSORNB-UHFFFAOYSA-N": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            ...
        }
        csv_path, txt_path, manifest_path = save_stock_files(
            stock=stock,
            stock_name="buyables-stock",
            output_dir=Path("data/1-benchmarks/stocks"),
            source_path=Path("data/1-benchmarks/raw-stocks/buyables-stock.txt"),
            statistics=stats
        )
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths (gzipped)
    csv_path = output_dir / f"{stock_name}.csv.gz"
    txt_path = output_dir / f"{stock_name}.txt.gz"
    manifest_path = output_dir / f"{stock_name}.manifest.json"

    logger.info(f"Saving stock files for '{stock_name}'...")

    # Sort by InChI key for deterministic output
    sorted_items = sorted(stock.items(), key=lambda x: x[0])

    # Save CSV (SMILES, InChIKey) - gzipped
    logger.debug(f"Writing CSV to {csv_path}...")
    try:
        with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "InChIKey"])  # Header
            for inchi_key, smiles in sorted_items:
                writer.writerow([smiles, inchi_key])
    except OSError as e:
        raise RetroCastIOError(f"Failed to write CSV file {csv_path}: {e}") from e

    # Save TXT (SMILES only, for models that need plain text) - gzipped
    logger.debug(f"Writing TXT to {txt_path}...")
    try:
        with gzip.open(txt_path, "wt", encoding="utf-8") as f:
            for _, smiles in sorted_items:
                f.write(f"{smiles}\n")
    except OSError as e:
        raise RetroCastIOError(f"Failed to write TXT file {txt_path}: {e}") from e

    # Create manifest
    logger.debug(f"Creating manifest at {manifest_path}...")

    sources: list[Path] = [source_path] if source_path else []

    manifest = create_manifest(
        action="canonicalize-stock",
        sources=sources,
        outputs=[
            (csv_path, stock, "stock"),  # CSV file with stock dict for content hashing
            (txt_path, stock, "stock"),  # TXT file (same content hash)
        ],
        root_dir=output_dir.parent.parent,  # Project root (2 levels up from stocks/)
        parameters={"stock_name": stock_name},
        statistics=statistics.to_manifest_dict() if statistics else {},
    )

    manifest_json = manifest.model_dump(mode="json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_json, f, indent=2, sort_keys=False)

    logger.info(f"✓ Saved {len(stock):,} molecules")
    logger.debug(f"  CSV: {csv_path}")
    logger.debug(f"  TXT: {txt_path}")
    logger.debug(f"  Manifest: {manifest_path}")

    return csv_path, txt_path, manifest_path


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
