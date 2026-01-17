import argparse
import importlib.resources
import sys
from pathlib import Path
from typing import Any

import yaml

from retrocast import __version__
from retrocast.cli import adhoc, handlers
from retrocast.paths import ENV_VAR_NAME, check_migration_needed, get_data_dir_source, resolve_data_dir
from retrocast.utils.logging import configure_script_logging, logger


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Loads configuration.
    Priority:
    1. Local file (config_path)
    2. Package default (src/retrocast/resources/default_config.yaml)
    """
    # 1. Try Local
    if config_path.exists():
        logger.debug(f"Loading local config from {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    # 2. Try Package Default
    try:
        logger.debug("Local config not found. Falling back to package defaults.")
        resource = importlib.resources.files("retrocast.resources").joinpath("default_config.yaml")
        with resource.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Could not load default configuration: {e}")
        sys.exit(1)


def main() -> None:
    configure_script_logging(use_rich=True)
    parser = argparse.ArgumentParser(
        description=f"Retrocast v{__version__}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("retrocast-config.yaml"), help="Path to config file")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"Override data directory (default: data/retrocast/, or {ENV_VAR_NAME} env var)",
    )
    parser.add_argument("--version", "-V", action="version", version=f"retrocast {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- CONFIG ---
    subparsers.add_parser("config", help="Show resolved configuration and paths")

    # --- INIT ---
    init_parser = subparsers.add_parser("init", help="Initialize a local configuration file")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config file")

    # --- ADAPT (Ad-Hoc) ---
    adapt_parser = subparsers.add_parser(
        "adapt", help="Convert raw predictions file to RetroCast schema (No config needed)"
    )
    adapt_parser.add_argument("--input", required=True, help="Path to raw predictions (.json.gz)")
    adapt_parser.add_argument("--output", required=True, help="Path to save processed routes (.json.gz)")
    adapt_parser.add_argument("--adapter", required=True, help="Name of the adapter to use (e.g. aizynth, dms)")
    adapt_parser.add_argument("--benchmark", help="Optional: Path to benchmark definition to ensure correct IDs")

    # --- LIST ---
    subparsers.add_parser("list", help="List configured models")

    # --- LIST ADAPTERS ---
    subparsers.add_parser("list-adapters", help="List all available adapters")

    # --- INFO ---
    info_parser = subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("--model", required=True)

    # --- INGEST ---
    ingest_parser = subparsers.add_parser("ingest", help="Process raw outputs")

    # Model selection
    m_group = ingest_parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("--model", help="Single model name")
    m_group.add_argument("--all-models", action="store_true", help="Process all models in config")

    # Dataset selection (Renamed to 'dataset' to match your old script habits, maps to 'benchmark')
    d_group = ingest_parser.add_mutually_exclusive_group(required=True)
    d_group.add_argument("--dataset", help="Single benchmark name")
    d_group.add_argument("--all-datasets", action="store_true", help="Process all available benchmarks")

    # Options
    ingest_parser.add_argument("--sampling-strategy", help="Override config sampling")
    ingest_parser.add_argument("--k", type=int, help="Override config k")
    ingest_parser.add_argument(
        "--anonymize", action="store_true", help="Hash the model name in the output folder (useful for blind review)"
    )
    ingest_parser.add_argument(
        "--ignore-stereo",
        action="store_true",
        help="Strip stereochemistry during SMILES canonicalization",
    )

    # --- SCORE ---
    score_parser = subparsers.add_parser("score", help="Run evaluation")
    # Model selection
    m_group_s = score_parser.add_mutually_exclusive_group(required=True)
    m_group_s.add_argument("--model", help="Single model name")
    m_group_s.add_argument("--all-models", action="store_true", help="Process all models")

    # Dataset selection
    d_group_s = score_parser.add_mutually_exclusive_group(required=True)
    d_group_s.add_argument("--dataset", help="Single benchmark name")
    d_group_s.add_argument("--all-datasets", action="store_true", help="Process all benchmarks")

    score_parser.add_argument("--stock", help="Override stock file name")
    score_parser.add_argument(
        "--ignore-stereo",
        action="store_true",
        help="Perform stereo-agnostic matching (drops stereochemistry from InChIKeys during scoring)",
    )

    # --- SCORE FILE (Ad-Hoc) ---
    # CHANGE: Add this new subparser block
    sf_parser = subparsers.add_parser("score-file", help="Run evaluation on specific files (adhoc mode)")
    sf_parser.add_argument("--benchmark", required=True, help="Path to benchmark .json.gz")
    sf_parser.add_argument("--routes", required=True, help="Path to predictions .json.gz")
    sf_parser.add_argument("--stock", required=True, help="Path to stock .txt")
    sf_parser.add_argument("--output", required=True, help="Path to output .json.gz")
    sf_parser.add_argument("--model-name", default="adhoc-model", help="Name of model for report")

    # --- ANALYZE ---
    analyze_parser = subparsers.add_parser("analyze", help="Generate reports")

    # Model selection
    m_group_a = analyze_parser.add_mutually_exclusive_group(required=True)
    m_group_a.add_argument("--model", help="Single model name")
    m_group_a.add_argument("--all-models", action="store_true", help="Process all models")

    # Dataset selection
    d_group_a = analyze_parser.add_mutually_exclusive_group(required=True)
    d_group_a.add_argument("--dataset", help="Single benchmark name")
    d_group_a.add_argument("--all-datasets", action="store_true", help="Process all benchmarks")

    analyze_parser.add_argument("--stock", help="Specific stock to analyze (optional, auto-detects if omitted)")
    analyze_parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate plots (requires 'viz' dependency group, e.g. uv run --extra viz, uv sync --extra viz, or uv pip install retrocast[viz]",
    )
    analyze_parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20, 50, 100],
        help="List of Top-K values to include in the markdown report (default: 1 3 5 10 20 50 100)",
    )

    # --- CREATE BENCHMARK ---
    create_bm_parser = subparsers.add_parser("create-benchmark", help="Create benchmark from SMILES list")
    create_bm_parser.add_argument("--input", required=True, help="Path to .txt or .csv")
    create_bm_parser.add_argument("--name", required=True, help="Name of the benchmark")
    create_bm_parser.add_argument("--output", required=True, help="Output path (.json.gz)")
    create_bm_parser.add_argument("--stock-name", required=True, help="Associated stock name")

    # --- VERIFY ---
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity and lineage")
    v_group = verify_parser.add_mutually_exclusive_group(required=True)
    v_group.add_argument("--target", help="Path to a specific manifest or directory")
    v_group.add_argument("--all", action="store_true", help="Verify all manifests in the data directory")
    verify_parser.add_argument("--deep", action="store_true", help="Perform deep verification of source files")
    verify_parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: missing files are failures (default: lenient mode treats missing files as warnings)",
    )

    args = parser.parse_args()

    # Commands that don't need config loading
    if args.command in ["init", "adapt", "score-file", "create-benchmark", "list-adapters"]:
        if args.command == "init":
            adhoc.handle_init(args)
        elif args.command == "adapt":
            adhoc.handle_adapt(args)
        elif args.command == "score-file":
            adhoc.handle_score_file(args)
        elif args.command == "create-benchmark":
            adhoc.handle_create_benchmark(args)
        elif args.command == "list-adapters":
            adhoc.handle_list_adapters(args)
        return

    # Load config (local or default)
    config = load_config(args.config)

    # Resolve data directory with priority: CLI > env > config > default
    cli_data_dir = getattr(args, "data_dir", None)
    config_data_dir = config.get("data_dir")
    resolved_data_dir = resolve_data_dir(cli_arg=cli_data_dir, config_value=config_data_dir)

    # Inject resolved data_dir into config for handlers
    config["data_dir"] = str(resolved_data_dir)
    config["_data_dir_source"] = get_data_dir_source(cli_arg=cli_data_dir, config_value=config_data_dir)

    # Check for migration and warn if needed
    migration_warning = check_migration_needed(resolved_data_dir)
    if migration_warning:
        logger.warning(f"Migration Notice: {migration_warning}")

    # Handle config command (needs resolved paths but not full config validation)
    if args.command == "config":
        handlers.handle_config(args, config)
        return

    try:
        if args.command == "list":
            handlers.handle_list(config)
        elif args.command == "info":
            handlers.handle_info(config, args.model)
        elif args.command == "ingest":
            handlers.handle_ingest(args, config)
        elif args.command == "score":
            handlers.handle_score(args, config)
        elif args.command == "analyze":
            handlers.handle_analyze(args, config)
        elif args.command == "verify":
            handlers.handle_verify(args, config)

    except Exception as e:
        logger.critical(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
