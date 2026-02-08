"""
Build custom building block indices for SynLlama from a stock file.

This script loads building blocks from a stock CSV file and builds
reaction-specific indices for use with SynLlama's reconstruction algorithm.
This is a one-time setup operation per stock file.

Example usage:
    uv run --directory scripts/planning/run-synllama 2-build-bb-indices.py --stock buyables-stock
    uv run --directory scripts/planning/run-synllama 2-build-bb-indices.py --stock buyables-stock --rxn-set 115rxns

The stock file should be located at: data/retrocast/0-assets/stocks/{stock_name}.csv.gz
Indices are saved to: synllama-data/inference/reconstruction/{rxn_set}/custom_bb_indices/{stock_name}/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from synllama.llm.build_custom_indices import build_custom_bb_indices

from retrocast.io import load_stock_file
from retrocast.paths import get_paths
from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()


@dataclass
class SynLlamaPaths:
    """Standard paths for SynLlama resources."""

    synllama_data: Path
    model_path: Path
    rxn_embedding_path: Path
    reaction_smarts_dict_path: Path
    custom_bb_base: Path
    stocks_dir: Path
    token_list_path: Path


def get_synllama_paths(rxn_set: str) -> SynLlamaPaths:
    """Get standard SynLlama paths using project root resolution.

    Args:
        rxn_set: Reaction set to use ("91rxns" or "115rxns").

    Returns:
        SynLlamaPaths with all standard resource paths.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "retrocast"
    paths = get_paths(data_dir)

    # synllama-data is in scripts/planning/run-synllama/
    synllama_data = Path(__file__).parent / "synllama-data"

    model_path = synllama_data / "inference" / "model" / f"SynLlama-1B-2M-{rxn_set}"
    rxn_embedding_path = synllama_data / "inference" / "reconstruction" / rxn_set / "rxn_embeddings"
    reaction_smarts_dict_path = rxn_embedding_path / "reaction_smarts_map.pkl"
    custom_bb_base = synllama_data / "inference" / "reconstruction" / rxn_set / "custom_bb_indices"
    token_list_path = synllama_data / "data" / "smiles_vocab.txt"

    return SynLlamaPaths(
        synllama_data=synllama_data,
        model_path=model_path,
        rxn_embedding_path=rxn_embedding_path,
        reaction_smarts_dict_path=reaction_smarts_dict_path,
        custom_bb_base=custom_bb_base,
        stocks_dir=paths["stocks"],
        token_list_path=token_list_path,
    )


def main() -> None:
    """Main entry point for building custom BB indices."""
    parser = argparse.ArgumentParser(description="Build custom BB indices for SynLlama from a stock file")
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Name of the stock file (without extension), e.g., 'buyables-stock'",
    )
    parser.add_argument(
        "--rxn-set",
        type=str,
        default="91rxns",
        choices=["91rxns", "115rxns"],
        help="Reaction set to use (default: 91rxns)",
    )
    args = parser.parse_args()

    # Get paths
    paths = get_synllama_paths(args.rxn_set)

    # Validate synllama-data exists
    if not paths.synllama_data.exists():
        raise FileNotFoundError(
            f"synllama-data directory not found at {paths.synllama_data}\n"
            "Please download and extract SynLlama model files to this location."
        )

    if not paths.reaction_smarts_dict_path.exists():
        raise FileNotFoundError(
            f"Reaction SMARTS dictionary not found at {paths.reaction_smarts_dict_path}\n"
            f"Please ensure the {args.rxn_set} reaction embeddings are available."
        )

    # Load building blocks
    stock_path = paths.stocks_dir / f"{args.stock}.csv.gz"
    building_blocks = list(load_stock_file(stock_path, return_as="smiles"))

    # Setup output directory
    output_dir = paths.custom_bb_base / args.stock
    if output_dir.exists():
        raise FileExistsError(
            f"BB indices already exist at {output_dir}\nIf you want to rebuild, please delete this directory first."
        )

    logger.info(f"Building BB indices for {args.stock} with {args.rxn_set}")
    logger.info(f"Output directory: {output_dir}")

    # Build indices
    try:
        custom_index_path = build_custom_bb_indices(
            custom_bbs=building_blocks,
            reaction_smarts_dict_path=str(paths.reaction_smarts_dict_path),
            output_dir=str(output_dir),
            token_list_path=str(paths.token_list_path) if paths.token_list_path.exists() else None,
        )

        logger.info(f"Successfully built custom BB indices at: {custom_index_path}")
        logger.info("These indices can now be used with 2-run-synllama.py")

    except Exception as e:
        logger.error(f"Failed to build BB indices: {e}", exc_info=True)
        # Clean up partial output if it exists
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)
            logger.info(f"Cleaned up partial output at {output_dir}")
        raise


if __name__ == "__main__":
    main()
