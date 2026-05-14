"""
pre-processes a flat completions file from an LLM batch into the retrocast-compatible format.

each input record (a JSON object with `meta.product_smiles` and `completion`) becomes
one entry in the per-target route list. the script groups records by target SMILES and
writes the result to a gzipped json file (`results.json.gz`).

the resulting json file is the expected input for the `llm-raw-answers` adapter.

supported input shapes:
  - a single JSON array on disk: `[{...}, {...}, ...]`
  - JSON Lines (one record per line)

---
example usage:
---
uv run scripts/llm-raw-answers/1-convert-to-json.py \
    --input completions.jsonl \
    --output data/evaluations/llm-raw-answers/some-benchmark
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from retrocast.utils.logging import logger


def _load_records(input_path: Path) -> list[dict]:
    """Load records from a JSON array file or JSONL file."""
    raw = input_path.read_text().strip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list of records")
        return data
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="preprocess llm completions into retrocast-compatible json.gz.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="path to completions file (.json or .jsonl).")
    parser.add_argument("-o", "--output", required=True, type=Path, help="path for the output directory.")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"input file not found at {args.input}")
        return

    output_dir = args.output
    results_path = output_dir / "results.json.gz"
    summary_path = output_dir / "summary.json"

    try:
        records = _load_records(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"error reading or parsing input file: {e}")
        return

    routes_by_target: dict[str, list[dict]] = defaultdict(list)
    skipped_no_target = 0
    for rec in records:
        meta = rec.get("meta") or {}
        target_smiles = meta.get("product_smiles")
        completion = rec.get("completion")
        if not target_smiles or not isinstance(completion, str):
            skipped_no_target += 1
            continue
        routes_by_target[target_smiles].append({"completion": completion})

    solved_count = len(routes_by_target)
    total_records = sum(len(v) for v in routes_by_target.values())
    logger.info(f"found {solved_count} unique targets across {total_records} completions.")
    if skipped_no_target:
        logger.warning(f"skipped {skipped_no_target} records without meta.product_smiles or completion.")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(routes_by_target, indent=2)
    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        f.write(json_str)
    logger.info(f"successfully wrote pre-processed data to {results_path}")

    summary_data = {
        "solved_count": solved_count,
        "total_records": total_records,
        "skipped_no_target": skipped_no_target,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
