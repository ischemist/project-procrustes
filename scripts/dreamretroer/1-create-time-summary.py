"""
Usage:
     uv run scripts/dreamretroer/1-create-time-summary.py
"""

import gzip
import json
from pathlib import Path

base_dir = Path("data/evaluations/dream-retroer")

for folder_path in base_dir.iterdir():
    if folder_path.is_dir():
        results_file = folder_path / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            total_targets = len(results)
            solved_count = sum(1 for v in results.values() if v.get("succ", False))
            time_elapsed = sum(v.get("time", 0) for v in results.values())

            summary = {"solved_count": solved_count, "total_targets": total_targets, "time_elapsed": time_elapsed}

            summary_file = folder_path / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4)

            gzipped_results_file = folder_path / "results.json.gz"
            with gzip.open(gzipped_results_file, "wt", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print(f"Processed {folder_path.name}: {summary}")
        else:
            print(f"results.json not found in {folder_path.name}")
