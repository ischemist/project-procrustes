"""
prepares raw ursa completions into the current retrocast-compatible results artifact.

this is a thin wrapper over `retrocast.adapters.ursa_llm_adapter.write_prepared_ursa_llm_results`.
the resulting `results.json.gz` file remains keyed by canonical target smiles so it can
flow through today's benchmark-centric `ingest` path unchanged.

supported input shapes:
  - `.json`
  - `.json.gz`
  - `.jsonl`
  - `.jsonl.gz`

example usage:

uv run scripts/ursa-llm/1-prepare-raw-results.py \
    --input completions.jsonl.gz \
    --output data/evaluations/ursa-llm/some-benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrocast.adapters.ursa_llm_adapter import write_prepared_ursa_llm_results
from retrocast.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare raw ursa completions into retrocast-compatible json.gz.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="path to completions file (.json, .json.gz, .jsonl, or .jsonl.gz).",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="path for the output directory.")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"input file not found at {args.input}")
        sys.exit(1)

    try:
        write_prepared_ursa_llm_results(input_path=args.input, output_dir=args.output)
    except (OSError, ValueError) as e:
        logger.error(f"error reading or parsing input file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
