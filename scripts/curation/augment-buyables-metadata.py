"""Augment buyables stock with commercial metadata from buyables_all.json.gz."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tqdm import tqdm

from retrocast.chem import get_inchi_key
from retrocast.io.blob import save_json_gz
from retrocast.io.data import load_stock_file
from retrocast.io.provenance import create_manifest
from retrocast.models.chem import BuyableMolecule, VendorSource
from retrocast.typing import InchiKeyStr, SmilesStr
from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging(log_level="INFO")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STOCKS_DIR = DATA_DIR / "1-benchmarks" / "stocks"
TMP_DIR = DATA_DIR / "tmp"


def load_buyables_metadata() -> dict[InchiKeyStr, dict]:
    """
    Load buyables_all.json.gz and create InchiKey → metadata mapping.

    Returns:
        Dict mapping InchiKey to metadata dict with keys: ppg, source, lead_time, link
    """
    buyables_path = TMP_DIR / "buyables_all.json.gz"
    logger.info(f"Loading buyables metadata from {buyables_path}...")

    metadata_map: dict[InchiKeyStr, dict] = {}
    failed_conversions = 0

    with gzip.open(buyables_path, "rt") as f:
        buyables_data = json.load(f)

    logger.info(f"Processing {len(buyables_data):,} buyables entries...")

    for entry in tqdm(buyables_data, desc="Converting to InchiKeys"):
        try:
            inchi_key = get_inchi_key(entry["smiles"])

            # Extract link from properties array
            link = None
            if entry.get("properties"):
                for prop in entry["properties"]:
                    if "link" in prop:
                        link = prop["link"]
                        break

            # Convert empty lead_time to None
            lead_time = entry["lead_time"] if entry["lead_time"] else None

            # Store metadata (prefer first occurrence if duplicates exist)
            if inchi_key not in metadata_map:
                metadata_map[inchi_key] = {
                    "ppg": entry["ppg"],
                    "source": entry["source"],
                    "lead_time": lead_time,
                    "link": link,
                }

        except Exception:
            failed_conversions += 1
            continue

    logger.info(f"Created metadata map with {len(metadata_map):,} unique InchiKeys")
    if failed_conversions > 0:
        logger.warning(f"Failed to convert {failed_conversions} SMILES to InchiKeys")

    return metadata_map


def augment_stock_with_metadata(
    stock_inchi_keys: set[InchiKeyStr],
    metadata_map: dict[InchiKeyStr, dict],
) -> list[BuyableMolecule]:
    """
    Augment stock InchiKeys with metadata from buyables_all.

    Args:
        stock_inchi_keys: Set of InchiKeys from buyables-stock.csv.gz
        metadata_map: Mapping from InchiKey to metadata dict

    Returns:
        List of BuyableMolecule objects with metadata
    """
    logger.info("Augmenting stock with metadata...")

    buyables = []
    missing_metadata = 0

    for inchi_key in tqdm(sorted(stock_inchi_keys), desc="Augmenting"):
        metadata = metadata_map.get(inchi_key)

        if metadata is None:
            missing_metadata += 1
            # Create entry without metadata
            buyables.append(
                BuyableMolecule(
                    smiles=SmilesStr(""),  # We don't have SMILES from stock file
                    inchikey=inchi_key,
                    ppg=None,
                    source=None,
                    lead_time=None,
                    link=None,
                )
            )
        else:
            # Map source string to enum
            source = VendorSource(metadata["source"]) if metadata["source"] else None

            buyables.append(
                BuyableMolecule(
                    smiles=SmilesStr(""),  # We don't have SMILES from stock file
                    inchikey=inchi_key,
                    ppg=metadata["ppg"],
                    source=source,
                    lead_time=metadata["lead_time"],
                    link=metadata["link"],
                )
            )

    logger.info(f"Created {len(buyables):,} augmented buyable entries")
    if missing_metadata > 0:
        logger.warning(
            f"{missing_metadata:,} molecules ({100 * missing_metadata / len(stock_inchi_keys):.1f}%) "
            f"had no metadata in buyables_all.json.gz"
        )

    return buyables


def main():
    """Main execution flow."""
    logger.info("=" * 80)
    logger.info("AUGMENT BUYABLES STOCK WITH METADATA")
    logger.info("=" * 80)

    # Load stock InchiKeys
    stock_path = STOCKS_DIR / "buyables-stock.csv.gz"
    stock_inchi_keys = load_stock_file(stock_path)
    logger.info(f"Loaded {len(stock_inchi_keys):,} molecules from {stock_path.name}")

    # Load metadata from buyables_all.json.gz
    metadata_map = load_buyables_metadata()

    # Calculate overlap statistics
    overlap = set(metadata_map.keys()) & stock_inchi_keys
    logger.info(
        f"Overlap: {len(overlap):,} molecules "
        f"({100 * len(overlap) / len(stock_inchi_keys):.1f}% of stock have metadata)"
    )

    # Augment stock with metadata
    buyables = augment_stock_with_metadata(stock_inchi_keys, metadata_map)

    # Save to buyables-meta.json.gz
    output_path = STOCKS_DIR / "buyables-meta.json.gz"
    manifest_path = STOCKS_DIR / "buyables-meta.manifest.json"

    logger.info(f"Saving augmented data to {output_path}...")

    # Convert to dict for JSON serialization
    buyables_dict = [b.model_dump(mode="json") for b in buyables]
    save_json_gz(buyables_dict, output_path)

    # Create manifest
    logger.info("Creating manifest...")
    sources = [stock_path, TMP_DIR / "buyables_all.json.gz"]

    # Calculate statistics
    with_metadata = sum(1 for b in buyables if b.ppg is not None)
    sources_dist = {}
    for b in buyables:
        if b.source:
            sources_dist[b.source.value] = sources_dist.get(b.source.value, 0) + 1

    statistics = {
        "total_molecules": len(buyables),
        "molecules_with_metadata": with_metadata,
        "molecules_without_metadata": len(buyables) - with_metadata,
        "metadata_coverage_pct": round(100 * with_metadata / len(buyables), 2),
        "source_distribution": sources_dist,
    }

    manifest = create_manifest(
        action="augment-buyables-metadata",
        sources=sources,
        outputs=[(output_path, buyables_dict, "unknown")],  # Use "unknown" for list of dicts
        root_dir=DATA_DIR,
        parameters={
            "stock_file": stock_path.name,
            "metadata_file": "buyables_all.json.gz",
        },
        statistics=statistics,
    )

    # Save manifest
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, sort_keys=False)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total molecules: {len(buyables):,}")
    logger.info(f"With metadata: {with_metadata:,} ({100 * with_metadata / len(buyables):.1f}%)")
    logger.info(f"Without metadata: {len(buyables) - with_metadata:,}")
    logger.info("\nSource distribution:")
    for source, count in sorted(sources_dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {source}: {count:,}")
    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
