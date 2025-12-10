import csv
import importlib.resources
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from retrocast.adapters import ADAPTER_MAP, get_adapter
from retrocast.api import score_predictions
from retrocast.curation.filtering import deduplicate_routes
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import load_benchmark, load_routes, save_routes
from retrocast.io.provenance import create_manifest
from retrocast.models.benchmark import create_benchmark, create_benchmark_target
from retrocast.models.chem import TargetInput

logger = logging.getLogger(__name__)


def handle_list_adapters(args: Any) -> None:
    """
    List all available adapters that can be used with the 'adapt' command.
    This command does not require a configuration file.
    """
    # Mapping of adapter names to their display names and format descriptions
    adapter_info = {
        "aizynth": ("AiZynthFinder", "bipartite graph"),
        "askcos": ("ASKCOS", "custom format"),
        "dms": ("DirectMultiStep", "recursive dict"),
        "dreamretro": ("DreamRetro", "precursor map"),
        "multistepttl": ("MultiStepTTL", "custom format"),
        "paroutes": ("PaRoutes", "reference format"),
        "retrochimera": ("RetroChimera", "precursor map"),
        "retrostar": ("Retro*", "precursor map"),
        "synllama": ("SynLlama", "precursor map"),
        "synplanner": ("SynPlanner", "bipartite graph"),
        "syntheseus": ("Syntheseus", "bipartite graph"),
    }

    print("Available adapters:")
    for name in sorted(ADAPTER_MAP.keys()):
        display_name, format_type = adapter_info.get(name, (name, "unknown format"))
        print(f"  - {name}: {display_name} ({format_type})")


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


def _process_csv_file(input_path: Path) -> dict[str, Any]:
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

            # Capture extra columns as metadata
            meta = {k: v for k, v in row.items() if k not in (id_col, smiles_col)}

            # Use official constructor (canonicalizes SMILES and computes InChIKey)
            targets[tid] = create_benchmark_target(
                id=tid,
                smiles=raw_smi,
                metadata=meta,
                acceptable_routes=[],  # Pure prediction task, no acceptable routes
            )

    return targets


def _process_txt_file(input_path: Path) -> dict[str, Any]:
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

        # Use official constructor (canonicalizes SMILES and computes InChIKey)
        targets[tid] = create_benchmark_target(
            id=tid,
            smiles=raw_smi,
            acceptable_routes=[],  # Pure prediction task
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

        # Create the BenchmarkSet with validation
        # Pure prediction tasks have no acceptable routes, so pass empty stock
        bm = create_benchmark(
            name=args.name,
            description=f"Created from {input_path.name}",
            stock=set(),  # Empty stock for pure prediction tasks
            stock_name=args.stock_name,
            targets=targets,
        )

        save_json_gz(bm, output_path)
        logger.info(f"Created benchmark '{args.name}' with {len(targets)} targets at {output_path}")

        # Create manifest
        manifest_path = args.output + ".manifest.json"
        manifest = create_manifest(
            action="[cli]create-benchmark",
            sources=[input_path],
            outputs=[(output_path, bm, "benchmark")],
            root_dir=output_path.parents[2],
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


def handle_init(args: Any) -> None:
    """
    Copies the internal default configuration to the current working directory.
    """
    target_path = Path("retrocast-config.yaml")

    if target_path.exists() and not args.force:
        logger.error(f"Configuration file '{target_path}' already exists. Use --force to overwrite.")
        sys.exit(1)

    try:
        # Load from package resources
        # 'retrocast.resources' must be a python package (have __init__.py)
        source = importlib.resources.files("retrocast.resources").joinpath("default_config.yaml")

        with importlib.resources.as_file(source) as src_path:
            shutil.copy(src_path, target_path)

        logger.info(f"Initialized configuration at [bold]{target_path.absolute()}[/]")
        logger.info("You can now edit this file to register custom models.")

    except Exception as e:
        logger.critical(f"Failed to initialize config: {e}")
        sys.exit(1)


def handle_adapt(args: Any) -> None:
    """
    Handler for 'retrocast adapt'.
    Converts a raw predictions file into the standardized RetroCast schema using a specific adapter.
    Does NOT require a full benchmark definition (infers targets from keys if needed).
    """
    input_path = Path(args.input)
    output_path = Path(args.output)
    adapter_name = args.adapter

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # 1. Load Resources
        adapter = get_adapter(adapter_name)
        raw_data = load_json_gz(input_path)  # Assume JSON/GZ for now, could expand

        # 2. Determine Targets
        # If benchmark is provided, use it as source of truth.
        # If not, iterate the raw_data keys and assume they are SMILES or IDs.
        targets_to_process = []

        if args.benchmark:
            benchmark_path = Path(args.benchmark)
            if not benchmark_path.exists():
                logger.error(f"Benchmark file not found: {benchmark_path}")
                sys.exit(1)
            bm = load_benchmark(benchmark_path)
            # Create (TargetInput, RawPayload) tuples
            for tid, target in bm.targets.items():
                payload = None
                if tid in raw_data:
                    payload = raw_data[tid]
                elif target.smiles in raw_data:
                    payload = raw_data[target.smiles]

                if payload:
                    targets_to_process.append((TargetInput(id=tid, smiles=target.smiles), payload))
        else:
            logger.info("No benchmark provided. Inferring targets from raw data keys.")
            if not isinstance(raw_data, dict):
                logger.error("Raw data must be a dictionary to infer targets (key=SMILES or ID).")
                sys.exit(1)

            for key, payload in raw_data.items():
                # We blindly assume the key is the SMILES for the TargetInput
                # The adapter will validate this against the internal structure usually
                # If key is an ID, this might fail strict SMILES validation in some adapters
                # but 'TargetInput' doesn't validate SMILES format strictly on init.
                targets_to_process.append((TargetInput(id=str(key), smiles=str(key)), payload))

        # 3. Processing Loop
        processed_routes = {}
        success_count = 0

        for target_input, payload in tqdm(targets_to_process, desc=f"Adapting ({adapter_name})"):
            try:
                routes = list(adapter.cast(payload, target=target_input))
                if routes:
                    unique = deduplicate_routes(routes)
                    processed_routes[target_input.id] = unique
                    success_count += 1
                else:
                    processed_routes[target_input.id] = []
            except Exception as e:
                logger.debug(f"Failed to adapt {target_input.id}: {e}")
                processed_routes[target_input.id] = []

        # 4. Save
        save_routes(processed_routes, output_path)
        logger.info(f"Adapted {success_count}/{len(targets_to_process)} targets. Saved to {output_path}")

        # 5. Manifest
        manifest_path = output_path.with_name(output_path.name + ".manifest.json")
        manifest = create_manifest(
            action="[cli]adapt",
            sources=[input_path],
            outputs=[(output_path, processed_routes, "predictions")],
            root_dir=output_path.parent,
            parameters={"adapter": adapter_name, "benchmark_provided": bool(args.benchmark)},
            statistics={"n_routes_saved": sum(len(r) for r in processed_routes.values())},
        )
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    except Exception as e:
        logger.critical(f"Adaptation failed: {e}", exc_info=True)
        sys.exit(1)
