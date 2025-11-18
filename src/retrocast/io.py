import csv
import gzip
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from tqdm import tqdm

from retrocast.domain.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException, RetroCastIOError, RetroCastSerializationError
from retrocast.schemas import EvaluationResults, Route, TargetInput
from retrocast.typing import SmilesStr
from retrocast.utils.logging import logger

# This allows us to return the same type that was passed in.
# e.g., load_model(MyModel) -> MyModel
T = TypeVar("T", bound=BaseModel)


def save_json_gz(data: dict[str, Any], path: Path) -> None:
    """
    Serializes a standard dictionary to a gzipped JSON file.

    The calling function is responsible for ensuring the dict is serializable
    (e.g., by calling .model_dump() on Pydantic objects first).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # We assume the data is a clean dict, no need for custom encoders.
        json_str = json.dumps(data, indent=2)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_str)
    except TypeError as e:
        logger.error(f"Data for {path} is not JSON serializable: {e}")
        raise RetroCastSerializationError(f"Data serialization error for {path}: {e}") from e
    except OSError as e:
        logger.error(f"Failed to write or serialize gzipped JSON to {path}: {e}")
        raise RetroCastIOError(f"Data saving/serialization error on {path}: {e}") from e


def load_json_gz(path: Path) -> dict[str, Any]:
    """
    Loads a gzipped JSON file into a Python dictionary.

    This is a low-level loader. It performs no validation beyond ensuring
    the file contains valid JSON.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            loaded_data = json.load(f)
            if not isinstance(loaded_data, dict):
                raise RetroCastIOError(f"Expected a JSON object (dict), but found {type(loaded_data)} in {path}")
            return loaded_data
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse gzipped JSON file: {path}")
        raise RetroCastIOError(f"Data loading error on {path}: {e}") from e


def save_routes(routes: dict[str, list[Route]], path: Path) -> None:
    """
    Saves a dictionary of routes to a gzipped JSON file.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        path: Path to the output gzipped JSON file.
    """
    data = {
        target_id: [route.model_dump(mode="json") for route in route_list] for target_id, route_list in routes.items()
    }
    save_json_gz(data, path)
    logger.info(f"Saved {sum(len(r) for r in routes.values())} routes for {len(routes)} targets to {path}")


def load_routes(path: Path) -> dict[str, list[Route]]:
    """
    Loads routes from a gzipped JSON file.

    Args:
        path: Path to the gzipped JSON file containing routes.

    Returns:
        Dictionary mapping target IDs to lists of Route objects.
    """
    data = load_json_gz(path)
    routes = {}
    for target_id, route_list in data.items():
        if not isinstance(route_list, list):
            raise RetroCastIOError(f"Expected list of routes for target '{target_id}', got {type(route_list)}")
        routes[target_id] = [Route.model_validate(r) for r in route_list]
    logger.info(f"Loaded {sum(len(r) for r in routes.values())} routes for {len(routes)} targets from {path}")
    return routes


def save_json(data: dict[str, Any], path: Path) -> None:
    """Saves a Python dictionary to a standard, uncompressed JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write to JSON file: {path}")
        raise RetroCastIOError(f"Data saving error on {path}: {e}") from e


def load_targets_csv(path: Path) -> dict[str, str]:
    """
    Loads a CSV file containing target IDs and SMILES.

    The CSV must have a header row with "Structure ID" and "SMILES" columns.
    Returns a dictionary mapping target IDs to SMILES strings.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logger.warning(f"CSV file {path} is empty or has no header")
                return {}
            data = {row["Structure ID"]: row["SMILES"] for row in reader}
            if not data:
                logger.warning(f"CSV file {path} is empty")
            return data
    except KeyError as e:
        logger.error(f"Missing required column in CSV file {path}: {e}")
        raise RetroCastIOError(f"CSV column {e} not found in {path}") from e
    except OSError as e:
        logger.error(f"Failed to read CSV file: {path}")
        raise RetroCastIOError(f"Data loading error on {path}: {e}") from e
    except csv.Error as e:
        logger.error(f"Failed to parse CSV file: {path}")
        raise RetroCastIOError(f"CSV parsing error on {path}: {e}") from e


def load_targets_json(path: Path) -> dict[str, str]:
    """
    Loads a JSON file containing target IDs and SMILES.

    Expected format: {"target_id_1": "SMILES_1", ...}
    Returns a dictionary mapping target IDs to SMILES strings.
    """
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                data = json.load(f)
        else:
            with path.open("r") as f:
                data = json.load(f)

        if not isinstance(data, dict):
            raise RetroCastIOError(f"Expected JSON object (dict), found {type(data)}")
        if not data:
            logger.warning(f"JSON file {path} is empty")
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read or parse JSON file: {path}")
        raise RetroCastIOError(f"Data loading error on {path}: {e}") from e


def load_target_ids(csv_path: str) -> list[tuple[int, str]]:
    """Load target IDs from CSV file with their position (1-based index)."""
    target_ids = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):  # 1-based indexing to match file numbering
                target_ids.append((idx, row["Structure ID"]))
    except OSError as e:
        logger.error(f"Failed to read CSV file: {csv_path}")
        raise RetroCastIOError(f"Data loading error on {csv_path}: {e}") from e
    except csv.Error as e:
        logger.error(f"Failed to parse CSV file: {csv_path}")
        raise RetroCastIOError(f"CSV parsing error on {csv_path}: {e}") from e
    return target_ids


def normalize_target_id_for_filename(target_id: str) -> str:
    """Convert target ID to the format used in filenames."""
    # Replace spaces, slashes, and special characters with underscores
    normalized = target_id.replace(" ", "_").replace("/", "_")
    # Handle parentheses - they become underscores in filenames
    normalized = normalized.replace("(", "_").replace(")", "_")
    # Clean up multiple underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def find_json_file_by_position(position: int, eval_dir: Path) -> Path | None:
    """Find the JSON file by position (file number)."""
    # Format position as 4-digit number with leading zeros (e.g., 0001, 0002, etc.)
    file_pattern = f"{position:04d}_*.json"
    for json_file in eval_dir.glob(file_pattern):
        return json_file
    return None


def find_json_file_by_name(target_id: str, eval_dir: Path, prefix: str = "*_") -> Path | None:
    """Find the JSON file corresponding to a target ID by name matching."""
    # First try exact match
    for json_file in eval_dir.glob(f"{prefix}{target_id}.json"):
        return json_file

    # If not found, try with normalized target ID (spaces -> underscores, etc.)
    normalized_id = normalize_target_id_for_filename(target_id)
    for json_file in eval_dir.glob(f"{prefix}{normalized_id}.json"):
        return json_file

    # For special cases like (R)-Crizotinib, try partial matching
    # Look for files that contain the main part of the target name
    if "(" in target_id and ")" in target_id:
        # Extract the main compound name (e.g., "Crizotinib" from "(R)-Crizotinib")
        main_name = target_id.split(")")[-1].strip("-").strip()
        for json_file in eval_dir.glob(f"{prefix}{main_name}*.json"):
            return json_file

    return None


def combine_evaluation_results(targets_csv: str, eval_dir: str, output_path: str, naming_convention: str) -> None:
    """Combine individual JSON results into a single file."""
    target_data = load_target_ids(targets_csv)
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        raise RetroCastIOError(f"Evaluation directory does not exist: {eval_dir}")

    results = {}
    missing_files = []

    with tqdm(total=len(target_data), desc="Combining results") as pbar:
        for position, target_id in target_data:
            pbar.set_description(f"Processing {target_id}")
            json_file = None

            if naming_convention == "askcos":
                # Try position-based matching first (most reliable)
                json_file = find_json_file_by_position(position, eval_path)
                # Fallback to name-based matching
                if json_file is None:
                    json_file = find_json_file_by_name(target_id, eval_path, prefix="*_")
            elif naming_convention == "chimera":
                # Chimera files are named directly as target_id.json
                json_file = find_json_file_by_name(target_id, eval_path, prefix="")
            else:
                raise RetroCastException(f"Unknown naming convention: {naming_convention}")

            if json_file is None:
                missing_files.append(target_id)
                pbar.update(1)
                continue

            try:
                with open(json_file) as f:
                    results[target_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                missing_files.append(target_id)
            pbar.update(1)

    # Write combined results
    with gzip.open(output_path, "wt") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Combined {len(results)} results into {output_path}")
    if missing_files:
        logger.warning(f"Missing files for {len(missing_files)} targets: {missing_files}")


def load_raw_routes(input_file: Path) -> list[dict]:
    """
    Load raw routes from a gzipped JSON file.

    This function loads a JSON file containing a list of route dictionaries,
    typically used as input for route conversion/curation scripts.

    Args:
        input_file: Path to the gzipped JSON file containing raw routes.

    Returns:
        List of dictionaries, where each dict represents a raw route.

    Raises:
        RetroCastIOError: If the file cannot be read or parsed.
    """
    logger.info(f"Loading raw routes from {input_file}...")
    try:
        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            all_routes = json.load(f)
        if not isinstance(all_routes, list):
            raise RetroCastIOError(f"Expected a list of routes in {input_file}, but got {type(all_routes)}")
        logger.info(f"Loaded {len(all_routes):,} total routes.")
        return all_routes
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse {input_file}: {e}")
        raise RetroCastIOError(f"Failed to load or parse {input_file}: {e}") from e


def load_and_prepare_targets(file_path: Path) -> dict[str, TargetInput]:
    """
    Loads a file containing target IDs and SMILES, canonicalizes the SMILES,
    and prepares a dictionary of TargetInput objects.
    """
    logger.info(f"Loading and preparing targets from {file_path}...")

    try:
        if file_path.suffix == ".csv":
            targets_raw = load_targets_csv(file_path)
        elif file_path.suffix == ".json" or (file_path.suffix == ".gz" and file_path.stem.endswith(".json")):
            targets_raw = load_targets_json(file_path)
        else:
            raise RetroCastException(f"Unsupported file format: {file_path}")
    except RetroCastIOError:
        # Let IO exceptions propagate with their specific type.
        raise

    prepared_targets: dict[str, TargetInput] = {}
    for target_id, raw_smiles in targets_raw.items():
        try:
            canon_smiles = canonicalize_smiles(raw_smiles)
            prepared_targets[target_id] = TargetInput(id=target_id, smiles=canon_smiles)
        except RetroCastException as e:
            msg = f"Invalid SMILES for target '{target_id}': {raw_smiles}. Cannot proceed."
            logger.error(msg)
            raise RetroCastException(msg) from e

    logger.info(f"Successfully prepared {len(prepared_targets)} targets.")
    return prepared_targets


def save_reaction_smiles(reactions: set[str] | list[str], path: Path) -> None:
    """
    Save reaction SMILES strings to a gzipped text file.

    Each reaction is written on its own line. The output is sorted
    for reproducibility.

    Args:
        reactions: Set or list of reaction SMILES strings.
        path: Path to the output .txt.gz file.
    """
    sorted_reactions = sorted(reactions)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            if sorted_reactions:
                f.write("\n".join(sorted_reactions))
                f.write("\n")
        logger.info(f"Saved {len(sorted_reactions)} reaction SMILES to {path}")
    except OSError as e:
        logger.error(f"Failed to write reaction SMILES to {path}: {e}")
        raise RetroCastIOError(f"Failed to save reaction SMILES to {path}: {e}") from e


def save_smiles_index(index: dict[str, str], path: Path) -> None:
    """
    Save a SMILES to target ID mapping as gzipped JSON.

    Args:
        index: Dictionary mapping SMILES strings to target IDs.
        path: Path to the output JSON file (can be .json or .json.gz).
    """
    if path.suffix == ".gz":
        save_json_gz(index, path)
    else:
        save_json(index, path)
    logger.info(f"Saved SMILES index with {len(index)} entries to {path}")


def load_smiles_index(path: Path) -> dict[str, TargetInput]:
    """
    Load a SMILES to target ID mapping from JSON.

    Args:
        path: Path to the JSON file containing the SMILES index (can be .json or .json.gz).

    Returns:
        Dictionary mapping SMILES strings to target IDs.

    Raises:
        RetroCastIOError: If the file cannot be read or parsed.
    """
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        if not isinstance(data, dict):
            raise RetroCastIOError(f"Expected JSON object (dict), found {type(data)} in {path}")
        logger.info(f"Loaded SMILES index with {len(data)} entries from {path}")

        converted_data = {smiles: TargetInput(id=target_id, smiles=smiles) for smiles, target_id in data.items()}

        return converted_data
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse SMILES index file: {path}")
        raise RetroCastIOError(f"SMILES index loading error on {path}: {e}") from e


def load_stock(path: Path) -> set[SmilesStr]:
    """
    Load stock molecules from a text file.

    Args:
        path: Path to the stock file (.txt). Each line contains one SMILES string.

    Returns:
        Set of SMILES strings (already canonicalized in the stock files).

    Raises:
        RetroCastIOError: If the file cannot be read.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            stock = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(stock)} molecules from stock file {path}")
        return stock
    except OSError as e:
        logger.error(f"Failed to read stock file: {path}")
        raise RetroCastIOError(f"Stock loading error on {path}: {e}") from e


def save_evaluation_results(results: EvaluationResults, path: Path) -> None:
    """
    Save evaluation results to a gzipped JSON file.

    Args:
        results: EvaluationResults object to save.
        path: Path to output .json.gz file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    data = results.model_dump(mode="json")
    save_json_gz(data, path)
    logger.info(f"Saved evaluation results to {path}")


def load_evaluation_results(path: Path) -> EvaluationResults:
    """
    Load evaluation results from a gzipped JSON file.

    Args:
        path: Path to the .json.gz file containing evaluation results.

    Returns:
        EvaluationResults object.

    Raises:
        RetroCastIOError: If the file cannot be read or parsed.
    """
    from retrocast.schemas import EvaluationResults

    data = load_json_gz(path)
    results = EvaluationResults.model_validate(data)
    logger.info(f"Loaded evaluation results from {path}")
    return results
