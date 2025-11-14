"""
Usage:
    uv run scripts/multistepttl/1-serialize-pickles.py --ds-name uspto-190
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from retrocast.exceptions import TtlRetroSerializationError
from retrocast.utils.logging import logger
from retrocast.utils.serializers import serialize_multistepttl_directory

base_dir = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="preprocess ttlretro pickled outputs into json.")
    parser.add_argument("--ds-name", type=Path, help="directory containing the target subdirectories with pickles.")
    args = parser.parse_args()

    all_serialized_data: dict[str, list[dict[str, Any]]] = {}

    res_folder = base_dir / "data" / "evaluations" / "multistep-ttl" / args.ds_name
    out_dir = base_dir / "data" / "evaluations" / "multistep-ttl" / args.ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for subdir in res_folder.iterdir():
        if not subdir.is_dir():
            continue

        target_name = subdir.name
        if target_name.startswith("USPTO"):
            target_name = target_name.replace("_", "/")
        logger.info(f"-> processing target: {target_name}")

        try:
            serialized_routes = serialize_multistepttl_directory(subdir)

            if serialized_routes is None:
                logger.warning(f"  - warning: could not find pickle files in {subdir}. skipping.")
                continue

            if not serialized_routes:
                logger.info(f"  - info: no solved routes found for {target_name}.")
                all_serialized_data[target_name] = []
                continue

            all_serialized_data[target_name] = serialized_routes
            logger.info(f"  - success: serialized {len(serialized_routes)} routes for {target_name}.")

        except TtlRetroSerializationError as e:
            logger.error(f"  - error: failed to process {target_name}. reason: {e}. skipping.")
        except Exception as e:
            logger.error(f"  - error: an unexpected error occurred for {target_name}: {e}. skipping.")

    # write the aggregated data to a single gzipped file
    logger.info(f"\nwriting aggregated data for {len(all_serialized_data)} targets")
    try:
        json_str = json.dumps(all_serialized_data, indent=2)
        with gzip.open(out_dir / "results.json.gz", "wt", encoding="utf-8") as f:
            f.write(json_str)
        logger.info("done.")
    except Exception as e:
        logger.error(f"error: failed to write output file. reason: {e}")


if __name__ == "__main__":
    main()
