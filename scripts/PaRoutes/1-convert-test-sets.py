"""
pre-processes the raw paroutes data into a benchmark-compatible format.

the raw data is a single json list of routes. this script converts it into
the standard ursa input format, which is a dictionary mapping a unique target id
to its raw route data.

it generates multiple versions: one with all routes, and several smaller
random samples for testing or lighter-weight analysis.

---
usage:
---
uv run scripts/PaRoutes/1-convert-test-sets.py
"""

import gzip
import json
import random
from pathlib import Path

from tqdm import tqdm

from ursa.adapters.paroutes_adapter import PaRoutesAdapter
from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo
from ursa.exceptions import UrsaException
from ursa.io import save_json_gz
from ursa.utils.logging import logger

# --- configuration ---
BASE_DIR = Path(__file__).resolve().parents[2]
PAROUTES_DIR = BASE_DIR / "data" / "paroutes"
test_sets = ["n1", "n5"]
# use `none` to signify processing the full dataset.
SAMPLE_SIZES = [None, 1000, 500]


def main() -> None:
    """main script execution."""

    for test_set in test_sets:
        input_file = PAROUTES_DIR / f"{test_set}-routes.json.gz"
        logger.info(f"loading raw routes from {input_file}...")
        try:
            with gzip.open(input_file, "rt", encoding="utf-8") as f:
                all_routes = json.load(f)
            if not isinstance(all_routes, list):
                logger.error(f"expected a list of routes in {input_file}, but got {type(all_routes)}")
                return
            logger.info(f"loaded {len(all_routes):,} total routes.")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"failed to load or parse {input_file}: {e}")
            return

        adapter = PaRoutesAdapter()

        for size in SAMPLE_SIZES:
            if size is None:
                sampled_routes = all_routes
                output_suffix = "full"
                logger.info("\n--- processing full dataset ---")
            else:
                if len(all_routes) < size:
                    logger.warning(
                        f"requested sample of {size} is larger than total routes ({len(all_routes)}). skipping."
                    )
                    continue
                sampled_routes = random.sample(all_routes, size)
                output_suffix = f"sample-{size}"
                logger.info(f"\n--- processing random sample of {size} ---")

            # this is the format ursa's main pipeline expects: a dict mapping id -> data
            processed_data = {}
            successful_routes = 0

            pbar = tqdm(enumerate(sampled_routes, 1), total=len(sampled_routes), desc="adapting routes")
            for i, raw_route in pbar:
                target_id = f"paroutes-n1-{i}"
                try:
                    # the adapter needs a targetinfo object to check for smiles mismatches.
                    target_smiles = canonicalize_smiles(raw_route["smiles"])
                    target_info = TargetInfo(id=target_id, smiles=target_smiles)

                    # the adapter yields valid benchmarktree objects.
                    # for paroutes, each input is a single route, so we expect 0 or 1 trees.
                    adapted_trees = list(adapter.adapt(raw_route, target_info))

                    if adapted_trees:
                        # we only save the first (and only) valid tree.
                        processed_data[target_id] = [tree.model_dump() for tree in adapted_trees]
                        successful_routes += 1

                except UrsaException as e:
                    logger.warning(f"could not process route {i} due to an error: {e}")
                except (KeyError, TypeError):
                    logger.warning(f"route {i} has invalid structure or missing 'smiles' key.")

            if not processed_data:
                logger.warning("no routes were successfully processed. no output file will be written.")
                continue

            output_filename = f"paroutes-{test_set}-{output_suffix}.json.gz"
            output_path = PAROUTES_DIR / output_filename
            logger.info(
                f"successfully adapted {successful_routes}/{len(sampled_routes)} routes. "
                f"saving to {output_path.relative_to(BASE_DIR)}..."
            )
            save_json_gz(processed_data, output_path)

    logger.info("\n--- finished preprocessing all paroutes samples. ---")


if __name__ == "__main__":
    main()
