import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from retrocast import __version__
from retrocast.cli import handlers
from retrocast.utils.logging import logger


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        # Fallback for dev environment
        dev_path = Path("retrocast-config.yaml")
        if dev_path.exists():
            config_path = dev_path
        else:
            logger.error(f"Config file not found at {config_path}")
            sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Retrocast v{__version__}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("retrocast-config.yaml"), help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- LIST ---
    subparsers.add_parser("list", help="List configured models")

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
    args = parser.parse_args()
    config = load_config(args.config)

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

    except Exception as e:
        logger.critical(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
