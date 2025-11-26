"""
Check that the canonicalization of the buyables stock is correct.

Usage:

    uv run scripts/dev/canonicalize-stock.py -i buyables-stock -o buyables-stock-export
    uv run scripts/dev/canonicalize-stock.py -i n1-stock -o n1-stock-export
    uv run scripts/dev/canonicalize-stock.py -i n5-stock -o n5-stock-export
    uv run scripts/dev/canonicalize-stock.py -i n1-n5-stock -o n1-n5-stock-export

"""

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import InvalidSmilesError, RetroCastException
from retrocast.utils.logging import configure_script_logging, logger

data_path = Path(__name__).resolve().parent / "data" / "1-benchmarks" / "stocks"

configure_script_logging()
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", help="Input file name")
argparser.add_argument("-o", "--output", help="Output file name")
args = argparser.parse_args()

stock_fname = args.input + ".txt"
save_fname = args.output + ".txt"

stock_lines = (data_path / stock_fname).read_text().splitlines()

old_smiles, canon_smiles = set(), set()
invalid_count = 0
pbar = tqdm(stock_lines, unit="smiles")
for line in pbar:
    smiles = line.strip()
    old_smiles.add(smiles)
    try:
        canon_smiles.add(canonicalize_smiles(smiles))
    except InvalidSmilesError:
        invalid_count += 1
    pbar.set_postfix({"canon_smi": len(canon_smiles), "invalid": invalid_count})

diff = old_smiles - canon_smiles
logger.info(f"{len(diff)} SMILES are not canonical")

with open(data_path / "export" / save_fname, "w") as f:
    f.write("SMILES,InChi Key\n")
    for canon_smi in sorted(canon_smiles):
        try:
            inchi_key = get_inchi_key(canon_smi)
            f.write(f"{canon_smi},{inchi_key}\n")
        except RetroCastException:
            logger.error(f"Failed to get InChI key for {canon_smi}")
