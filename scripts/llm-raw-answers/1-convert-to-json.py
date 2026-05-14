"""
pre-processes a flat completions file from an LLM batch into the retrocast-compatible format.

each input record (a JSON object with `meta.product_smiles` and `completion`) becomes
one entry in the per-target route list. the script canonicalizes each target smiles,
groups records by canonical target smiles, and writes the result to a gzipped json file
(`results.json.gz`).

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

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import RetroCastException
from retrocast.utils.logging import logger


def _load_records(input_path: Path) -> list[Any]:
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
        sys.exit(1)

    output_dir = args.output
    results_path = output_dir / "results.json.gz"
    summary_path = output_dir / "summary.json"

    try:
        records = _load_records(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"error reading or parsing input file: {e}")
        sys.exit(1)

    routes_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped_records = 0
    accepted_records = 0
    for rec in records:
        if not isinstance(rec, dict):
            skipped_records += 1
            continue
        meta = rec.get("meta") or {}
        if not isinstance(meta, dict):
            skipped_records += 1
            continue
        target_smiles = meta.get("product_smiles")
        completion = rec.get("completion")
        if not target_smiles or not isinstance(completion, str):
            skipped_records += 1
            continue
        try:
            canonical_target = canonicalize_smiles(target_smiles)
        except RetroCastException:
            skipped_records += 1
            continue
        routes_by_target[canonical_target].append({"completion": completion})
        accepted_records += 1

    solved_count = len(routes_by_target)
    total_records = len(records)
    logger.info(f"found {solved_count} unique targets across {accepted_records} accepted completions.")
    if skipped_records:
        logger.warning(f"skipped {skipped_records} invalid or incomplete records.")

    output_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        json.dump(routes_by_target, f, indent=2)
    logger.info(f"successfully wrote pre-processed data to {results_path}")

    summary_data = {
        "solved_count": solved_count,
        "total_records": total_records,
        "accepted_records": accepted_records,
        "skipped_records": skipped_records,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
