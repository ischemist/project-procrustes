import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from retrocast import __version__
from retrocast.cli import handlers
from retrocast.utils.logging import logger


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads the main yaml configuration file."""
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Retrocast v{__version__}: The Retrosynthesis Benchmark Platform",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("retrocast-config.yaml"), help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- INGEST ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw model outputs into standard format.")
    ingest_parser.add_argument("--model", required=True, help="Model name (must match config)")
    ingest_parser.add_argument("--benchmark", required=True, help="Benchmark set name (e.g. stratified-linear-600)")
    ingest_parser.add_argument("--raw-file", required=True, type=Path, help="Path to raw output file")
    # Optional Sampling overrides
    ingest_parser.add_argument("--sampling", choices=["top-k", "random-k", "by-depth"], help="Apply sampling strategy")
    ingest_parser.add_argument("--k", type=int, help="K value for sampling")
    ingest_parser.add_argument("--no-anonymize", action="store_true", help="Use model name instead of hash for folder")

    # --- SCORE ---
    score_parser = subparsers.add_parser("score", help="Evaluate processed routes against stock/GT.")
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--benchmark", required=True)
    score_parser.add_argument("--stock", help="Override stock name defined in benchmark")

    # --- ANALYZE ---
    analyze_parser = subparsers.add_parser("analyze", help="Generate statistical reports.")
    analyze_parser.add_argument("--model", required=True)
    analyze_parser.add_argument("--benchmark", required=True)
    analyze_parser.add_argument("--compare-with", help="Another model name to run paired comparisons against")

    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)

    # Dispatch
    try:
        if args.command == "ingest":
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
