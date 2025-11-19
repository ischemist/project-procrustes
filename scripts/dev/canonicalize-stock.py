"""
Check that the canonicalization of the buyables stock is correct.

Usage:

uv run scripts/dev/canonicalize-stock.py -i retrocast-bb-stock-v3 -o retrocast-bb-stock-v3-canon

"""

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import InvalidSmilesError

data_path = Path(__name__).resolve().parent / "data" / "models" / "assets"
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", help="Input file name")
argparser.add_argument("-o", "--output", help="Output file name")
args = argparser.parse_args()

stock_fname = args.input + ".csv"
save_fname = args.output + ".csv"

stock_lines = (data_path / stock_fname).read_text().splitlines()

old_smi = set()
canon_smi = set()
invalid = set()
pbar = tqdm(stock_lines, unit="smiles")
for line in pbar:
    smiles = line.split(",")[1]
    old_smi.add(smiles)
    try:
        canon_smi.add(canonicalize_smiles(smiles))
    except InvalidSmilesError:
        invalid.add(smiles)
    pbar.set_postfix({"canon_smi": len(canon_smi), "invalid": len(invalid)})

diff = old_smi - canon_smi
print(f"{len(diff)} SMILES are not canonical")

with open(data_path / save_fname, "w") as f:
    f.write("\n".join(sorted(canon_smi)))
