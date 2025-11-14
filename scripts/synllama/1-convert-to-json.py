"""
pre-processes raw csv output from the synllama model into the retrocast-compatible format.

this script is a one-off etl step required because synllama outputs a flat csv
where each row is a single route, unlike the nested json format used by most other
models. it reads this csv, groups routes by the target's 'structure id', and
writes the result to a gzipped json file (`results.json.gz`).

the resulting json file is the expected input for the `synllamaadapter`.

---
example usage:
---
uv run scripts/synllama/1-convert-to-json.py \
    --input data/evaluations/synllama/uspto-190/results.csv \
    --output data/evaluations/synllama/uspto-190

uv run scripts/synllama/1-convert-to-json.py \
    -i tests/testing_data/model-predictions/synllama/targets.csv \
    -o tests/testing_data/model-predictions/synllama
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from retrocast.utils.logging import logger


def main() -> None:
    """converts synllama csv output to retrocast-compatible gzipped json."""
    parser = argparse.ArgumentParser(description="preprocess synllama csv output.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="path to the input csv file.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="path for the output directory.")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"input file not found at {args.input}")
        return

    output_dir = args.output
    results_path = output_dir / "results.json.gz"
    summary_path = output_dir / "summary.json"

    # we group by 'structure id' as this is the canonical name for the target.
    # each row in the csv represents a single route.
    routes_by_target = defaultdict(list)
    total_time_s = 0.0
    try:
        with args.input.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                target_id = row.get("Structure ID")
                synthesis_str = row.get("synthesis")
                if target_id and synthesis_str:
                    # the adapter will only need the 'synthesis' string.
                    # we wrap it in a dict to match the "list of route objects" pattern.
                    routes_by_target[target_id].append({"synthesis_string": synthesis_str})
                    time_str = row.get("time, s")
                    if time_str:
                        try:
                            total_time_s += float(time_str)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"could not parse time value '{time_str}' for target '{target_id}'. skipping."
                            )
    except (OSError, csv.Error) as e:
        logger.error(f"error reading or parsing csv file: {e}")
        return

    solved_count = len(routes_by_target)
    logger.info(f"found {solved_count} unique targets with routes.")
    logger.info(f"total compute time from csv: {total_time_s:.2f} seconds.")

    # create output directory and save files
    output_dir.mkdir(parents=True, exist_ok=True)

    # save main results
    json_str = json.dumps(routes_by_target, indent=2)
    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        f.write(json_str)
    logger.info(f"successfully wrote pre-processed data to {results_path}")

    # save summary
    summary_data = {"solved_count": solved_count, "time_elapsed": total_time_s}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
