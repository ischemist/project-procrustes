from pathlib import Path

from retrocast.io import load_stock_file
from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()

base_dir = Path(__file__).resolve().parents[2]
stock_dir = base_dir / "data" / "1-benchmarks" / "stocks"

buyables = load_stock_file(stock_dir / "buyables-stock.csv.gz")
# eMolecules = load_stock_file(stock_dir / "eMolecules.csv.gz")
n1 = load_stock_file(stock_dir / "n1-stock.csv.gz")
n5 = load_stock_file(stock_dir / "n5-stock.csv.gz")
# ursa = load_stock_file(stock_dir / "ursa-bb-stock-v3-canon.csv.gz")

# is n1 and n5 in eMolecules?
# logger.info(f"n1 & eMolecules: {len(n1 & eMolecules)}")
# logger.info(f"n5 & eMolecules: {len(n5 & eMolecules)}")
logger.info(f"n1 & n5: {len(n1 & n5)}")

# is n1 and n5 in buyables?
logger.info(f"n1 & buyables: {len(n1 & buyables)}")
logger.info(f"n5 & buyables: {len(n5 & buyables)}")

# is n1 and n5 in ursa?
# logger.info(f"n1 & ursa: {len(n1 & ursa)}")
# logger.info(f"n5 & ursa: {len(n5 & ursa)}")

# # write unique n1 & n5 to n1-n5-stock.txt
# with open(stock_dir / "n1-n5-stock.txt", "w") as f:
#     for smiles in n1 | n5:
#         f.write(smiles + "\n")
