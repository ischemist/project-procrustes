import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retrocast.api import score_predictions
from retrocast.chem import (
    canonicalize_smiles,
)
from retrocast.io.blob import save_json_gz
from retrocast.io.data import load_benchmark, load_routes
from retrocast.io.provenance import create_manifest
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget

logger = logging.getLogger(__name__)


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    """
    Find a column by trying candidate names (case-insensitive match).

    Args:
        fieldnames: Available column names from the CSV
        candidates: List of acceptable column names to search for

    Returns:
        The actual column name from fieldnames, or None if not found
    """
    lower_names = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_names:
            return lower_names[candidate.lower()]
    return None


def _process_csv_file(input_path: Path) -> dict[str, BenchmarkTarget]:
    """
    Process a CSV file and extract benchmark targets.

    Args:
        input_path: Path to the CSV file

    Returns:
        Dictionary mapping target IDs to BenchmarkTarget objects

    Raises:
        ValueError: If required columns are missing or malformed
    """
    targets = {}

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no header row.")

        # Find column names flexibly
        smiles_col = _find_column(reader.fieldnames, ["smiles", "smi", "SMILES", "SMI", "structure"])
        id_col = _find_column(reader.fieldnames, ["id", "target_id", "structure_id", "ID", "Target ID", "Structure ID"])

        if not smiles_col:
            raise ValueError(
                f"CSV must contain a SMILES column. Available columns: {', '.join(reader.fieldnames)}. "
                f"Acceptable names: smiles, smi, SMILES, SMI, structure"
            )
        if not id_col:
            raise ValueError(
                f"CSV must contain an ID column. Available columns: {', '.join(reader.fieldnames)}. "
                f"Acceptable names: id, target_id, structure_id, ID, Target ID, Structure ID"
            )

        for row in reader:
            tid = row[id_col].strip()
            raw_smi = row[smiles_col].strip()

            # Basic Validation & Calculation
            canon_smi = canonicalize_smiles(raw_smi)

            # Capture extra columns as metadata
            meta = {k: v for k, v in row.items() if k not in (id_col, smiles_col)}

            targets[tid] = BenchmarkTarget(
                id=tid, smiles=canon_smi, metadata=meta, ground_truth=None, is_convergent=None, route_length=None
            )

    return targets


def _process_txt_file(input_path: Path) -> dict[str, BenchmarkTarget]:
    """
    Process a TXT file (one SMILES per line) and extract benchmark targets.
    Auto-generates sequential IDs.

    Args:
        input_path: Path to the TXT file

    Returns:
        Dictionary mapping target IDs to BenchmarkTarget objects
    """
    targets = {}

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    width = len(str(len(lines)))
    for i, raw_smi in enumerate(lines):
        tid = f"target-{i + 1:0{width}d}"
        canon_smi = canonicalize_smiles(raw_smi)

        targets[tid] = BenchmarkTarget(
            id=tid, smiles=canon_smi, ground_truth=None, is_convergent=None, route_length=None
        )

    return targets


def handle_create_benchmark(args: Any) -> None:
    """
    Creates a BenchmarkSet from a simple input file (TXT or CSV).
    Does not require ground truth routes.
    """
    input_path = Path(args.input)
    output_path = Path(args.output + ".json.gz")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Dispatch to appropriate processor based on file type
        if input_path.suffix == ".csv":
            targets = _process_csv_file(input_path)
        elif input_path.suffix == ".txt":
            targets = _process_txt_file(input_path)
        else:
            logger.error("Unsupported file extension. Use .csv or .txt")
            sys.exit(1)

        # Create the BenchmarkSet
        bm = BenchmarkSet(
            name=args.name, description=f"Created from {input_path.name}", stock_name=args.stock_name, targets=targets
        )

        save_json_gz(bm, output_path)
        logger.info(f"Created benchmark '{args.name}' with {len(targets)} targets at {output_path}")

        # Create manifest
        manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
        manifest = create_manifest(
            action="[cli]create-benchmark",
            sources=[input_path],
            outputs=[(output_path, bm)],
            parameters={"name": args.name, "stock_name": args.stock_name},
            statistics={"n_targets": len(targets)},
        )

        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info(f"Created manifest at {manifest_path}")

    except Exception as e:
        logger.critical(f"Failed to create benchmark: {e}", exc_info=True)
        sys.exit(1)


def handle_score_file(args: Any) -> None:
    """
    Handler for 'retrocast score-file'.
    Scores predictions from a specific file against a specific benchmark file.
    """
    benchmark_path = Path(args.benchmark)
    routes_path = Path(args.routes)
    stock_path = Path(args.stock)
    output_path = Path(args.output)

    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    if not routes_path.exists():
        logger.error(f"Routes file not found: {routes_path}")
        sys.exit(1)
    if not stock_path.exists():
        logger.error(f"Stock file not found: {stock_path}")
        sys.exit(1)

    try:
        # Load inputs
        benchmark = load_benchmark(benchmark_path)
        routes = load_routes(routes_path)

        # Run Scoring via API
        results = score_predictions(
            benchmark=benchmark,
            predictions=routes,
            stock=stock_path,
            model_name=args.model_name,
        )

        # Save
        save_json_gz(results, output_path)
        logger.info(f"Scoring complete. Results saved to {output_path}")

    except Exception as e:
        logger.critical(f"Scoring failed: {e}", exc_info=True)
        sys.exit(1)
