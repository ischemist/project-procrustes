"""
Combine DMS prediction results from multiple parts into a single folder.

This script takes results from subparts (e.g., uspto-190-pt1, uspto-190-pt2, etc.)
and combines them into a single folder (e.g., uspto-190).

Example Usage:
    uv run scripts/directmultistep/3-combine-results.py dms-wide-fp16 uspto-190
"""

import argparse
import json
from pathlib import Path

from retrocast.io import load_json_gz, save_json_gz


def combine_results(parent_dir: Path, base_name: str, parts: list[str]) -> None:
    """
    Combine results from multiple parts into a single folder.

    Args:
        parent_dir: Parent directory containing the part folders (e.g., "model-name-eval")
        base_name: Base name for the combined folder (e.g., "uspto-190")
        parts: List of part suffixes (e.g., ["pt1", "pt2", "pt3", "pt4"])
    """

    # Create combined folder
    combined_dir = parent_dir / base_name
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Initialize combined data structures
    combined_valid_results = {}
    combined_ursa_bb_results = {}
    # Initialize counters
    total_raw_solved = 0
    total_ursa_bb_solved = 0
    total_time_elapsed = 0.0

    print(f"Combining results for {base_name} from {len(parts)} parts...")

    # Process each part
    for part in parts:
        part_dir = parent_dir / f"{base_name}-{part}"

        if not part_dir.exists():
            print(f"Warning: Directory {part_dir} does not exist, skipping...")
            continue

        print(f"Processing {part_dir}...")

        # Load results.json
        results_file = part_dir / "summary.json"
        if results_file.exists():
            with open(results_file) as f:
                part_results = json.load(f)
                total_raw_solved += part_results.get("raw_solved_count", 0)
                total_time_elapsed += part_results.get("time_elapsed", 0.0)
                total_ursa_bb_solved += part_results.get("retrocast_bb_solved_count", 0)
        # Load valid_results
        valid_file = part_dir / "valid_results.json.gz"
        if valid_file.exists():
            part_valid = load_json_gz(valid_file)
            combined_valid_results.update(part_valid)

        # Load ursa_bb_results
        ursa_bb_file = part_dir / "ursa_bb_results.json.gz"
        if ursa_bb_file.exists():
            part_ursa_bb = load_json_gz(ursa_bb_file)
            combined_ursa_bb_results.update(part_ursa_bb)

    # Save combined results
    combined_results = {
        "raw_solved_count": total_raw_solved,
        "retrocast_bb_solved_count": total_ursa_bb_solved,
        "time_elapsed": total_time_elapsed,
        "parts_combined": len(parts),
    }

    # Save results.json
    with open(combined_dir / "summary.json", "w") as f:
        json.dump(combined_results, f, indent=2)

    # Save combined data files
    save_json_gz(combined_valid_results, combined_dir / "valid_results.json.gz")
    save_json_gz(combined_ursa_bb_results, combined_dir / "ursa_bb_results.json.gz")

    print(f"\nCombined results saved to: {combined_dir}")
    print(f"Total targets processed: {len(combined_valid_results)}")
    print(f"Raw solved count: {total_raw_solved}")
    print(f"RetroCast BB solved count: {total_ursa_bb_solved}")
    print(f"Total time elapsed: {total_time_elapsed:.2f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine DMS prediction results from multiple parts")
    parser.add_argument("parent_dir", help="Parent directory containing the part folders (e.g., 'model-name-eval')")
    parser.add_argument("base_name", help="Base name for the combined folder (e.g., 'uspto-190')")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["pt1", "pt2", "pt3", "pt4"],
        help="List of part suffixes to combine (default: pt1 pt2 pt3 pt4)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "evaluations",
        help="Base directory containing the parent folder",
    )

    args = parser.parse_args()

    parent_path = args.base_dir / args.parent_dir
    combine_results(parent_path, args.base_name, args.parts)


if __name__ == "__main__":
    main()
