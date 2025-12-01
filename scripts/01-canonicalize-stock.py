"""
Production-quality stock canonicalization script.

Reads raw stock files (one SMILES per line), canonicalizes SMILES, deduplicates by InChI key,
and outputs dual format: CSV (SMILES,InChIKey) and TXT (SMILES only) with manifest tracking.

Usage:
    uv run scripts/01-canonicalize-stock.py --input buyables-stock --output buyables-stock
    uv run scripts/01-canonicalize-stock.py --input n1-stock --output n1-stock
    uv run scripts/01-canonicalize-stock.py --input n5-stock --output n5-stock
    uv run scripts/01-canonicalize-stock.py --input n1-n5-stock --output n1-n5-stock
    uv run scripts/01-canonicalize-stock.py --input eMolecules --output emolecules-stock

Input files should be in: data/1-benchmarks/raw-stocks/
Output files written to: data/1-benchmarks/stocks/

Outputs:
    - {output}.csv: SMILES,InChIKey format with header
    - {output}.txt: Canonical SMILES only, one per line
    - {output}.manifest.json: Provenance tracking with hashes and statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import RetroCastException
from retrocast.io.data import save_stock_files
from retrocast.models.chem import StockStatistics
from retrocast.typing import InchiKeyStr, SmilesStr
from retrocast.utils.logging import configure_script_logging, logger

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_STOCKS_DIR = PROJECT_ROOT / "data" / "1-benchmarks" / "raw-stocks"
OUTPUT_STOCKS_DIR = PROJECT_ROOT / "data" / "1-benchmarks" / "stocks"


def canonicalize_stock(
    input_path: Path,
) -> tuple[dict[InchiKeyStr, SmilesStr], StockStatistics]:
    """
    Reads raw stock file, canonicalizes SMILES, and deduplicates by InChI key.

    Uses streaming to handle arbitrarily large files with minimal memory footprint.

    Args:
        input_path: Path to raw stock file (one SMILES per line)

    Returns:
        Tuple of (stock_dict, statistics)
        - stock_dict: mapping of InChIKey -> canonical SMILES
        - statistics: StockStatistics with processing metrics
    """
    logger.info(f"Streaming raw stock from {input_path}...")

    # Statistics tracking (we'll count lines as we go)
    stats = StockStatistics(raw_input_lines=0)

    # Process each line
    stock_dict: dict[InchiKeyStr, SmilesStr] = {}
    smiles_to_inchi: dict[SmilesStr, InchiKeyStr] = {}  # Track for duplicate SMILES detection

    # Stream file line-by-line (OSError will propagate naturally if file doesn't exist)
    with input_path.open("r", encoding="utf-8") as f:
        # No total for tqdm, but still get throughput and count
        pbar = tqdm(f, unit="lines", desc="Canonicalizing")

        for line_num, line in enumerate(pbar, start=1):
            stats.raw_input_lines += 1
            smiles = line.strip()

            # Skip empty lines
            if not smiles:
                stats.empty_lines += 1
                continue

            # Skip header if present (CSV files)
            if smiles.upper().startswith("SMILES") or smiles.upper().startswith("INCHI"):
                stats.empty_lines += 1
                continue

            # Canonicalize SMILES
            try:
                canon_smiles = canonicalize_smiles(smiles)
            except RetroCastException as e:  # Covers InvalidSmilesError (subclass)
                stats.invalid_smiles += 1
                logger.debug(f"Line {line_num}: Failed to canonicalize '{smiles}': {e}")
                pbar.set_postfix({"invalid": stats.invalid_smiles, "unique": len(stock_dict)})
                continue

            # Generate InChI key
            try:
                inchi_key = get_inchi_key(canon_smiles)
            except RetroCastException as e:  # Covers InvalidSmilesError (subclass)
                stats.inchi_generation_failed += 1
                logger.debug(f"Line {line_num}: Failed to generate InChI for '{canon_smiles}': {e}")
                pbar.set_postfix(
                    {
                        "inchi_fail": stats.inchi_generation_failed,
                        "unique": len(stock_dict),
                    }
                )
                continue

            # Track SMILES duplicates (same SMILES, different representation)
            if canon_smiles in smiles_to_inchi:
                stats.duplicate_smiles += 1
                continue

            # Track InChI duplicates (tautomers, stereoisomers, etc.)
            if inchi_key in stock_dict:
                stats.duplicate_inchikeys += 1
                # Keep the first SMILES we saw for this InChI
                continue

            # Add to stock
            stock_dict[inchi_key] = canon_smiles
            smiles_to_inchi[canon_smiles] = inchi_key

            pbar.set_postfix(
                {
                    "unique": len(stock_dict),
                    "dup_inchi": stats.duplicate_inchikeys,
                }
            )

    stats.unique_molecules = len(stock_dict)

    logger.info("✓ Canonicalization complete")
    logger.info(f"  Raw input lines: {stats.raw_input_lines:,}")
    logger.info(f"  Empty lines: {stats.empty_lines:,}")
    logger.info(f"  Invalid SMILES: {stats.invalid_smiles:,}")
    logger.info(f"  InChI generation failed: {stats.inchi_generation_failed:,}")
    logger.info(f"  Duplicate SMILES: {stats.duplicate_smiles:,}")
    logger.info(f"  Duplicate InChI keys: {stats.duplicate_inchikeys:,}")
    logger.info(f"  → Unique molecules: {stats.unique_molecules:,}")

    return stock_dict, stats


def main():
    configure_script_logging()

    parser = argparse.ArgumentParser(
        description="Canonicalize and deduplicate stock files by InChI key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file name (without .txt extension, e.g., 'buyables-stock')",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file name (without extension, e.g., 'buyables-stock')",
    )
    args = parser.parse_args()

    # Construct paths
    input_fname = args.input if args.input.endswith(".txt") else f"{args.input}.txt"
    input_path = RAW_STOCKS_DIR / input_fname

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Canonicalize and deduplicate
    stock_dict, stats = canonicalize_stock(input_path)

    # Save outputs
    logger.info(f"Saving outputs to {OUTPUT_STOCKS_DIR}...")
    csv_path, txt_path, manifest_path = save_stock_files(
        stock=stock_dict,
        stock_name=args.output,
        output_dir=OUTPUT_STOCKS_DIR,
        source_path=input_path,
        statistics=stats,
    )

    logger.info("✓ Stock canonicalization complete!")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  TXT: {txt_path}")
    logger.info(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
