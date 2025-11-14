"""
Combine RetroChimera evaluation results into a single JSON file.

This script reads individual JSON files from the RetroChimera evaluation directory
and combines them into a single results.json.gz file, mapping target IDs to their results.

Usage:

uv run python scripts/retrochimera/1-gather-chimera-results.py \
  --targets-csv data/targets/uspto-190.csv \
  --eval-dir data/evaluations/retrochimera/uspto-190 \
  --output data/evaluations/retrochimera/uspto-190/results.json.gz
"""

import argparse

from retrocast.io import combine_evaluation_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine RetroChimera evaluation results")
    parser.add_argument("--targets-csv", required=True, help="Path to CSV file with target IDs")
    parser.add_argument("--eval-dir", required=True, help="Directory containing individual JSON result files")
    parser.add_argument("--output", required=True, help="Output path for combined results.json.gz")

    args = parser.parse_args()

    combine_evaluation_results(args.targets_csv, args.eval_dir, args.output, naming_convention="chimera")


if __name__ == "__main__":
    main()
