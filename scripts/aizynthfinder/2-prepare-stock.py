""" Module containing classes and routines for making stock input to the tree search.

Usage:
    uv run --extra aizyn scripts/aizynthfinder/2-prepare-stock.py \
        --files data/1-benchmarks/stocks/n1-n5-stock.txt \
        --source plain \
        --output data/1-benchmarks/stocks/n1-n5-stock.hdf5 \
        --target hdf5
    uv run --extra aizyn scripts/aizynthfinder/2-prepare-stock.py \
        --files data/1-benchmarks/stocks/buyables-stock.txt \
        --source plain \
        --output data/1-benchmarks/stocks/buybles-stock.hdf5 \
        --target hdf5

"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterable

import pandas as pd
from aizynthfinder.chem import Molecule, MoleculeException


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("smiles2stock")
    parser.add_argument(
        "--files",
        required=True,
        nargs="+",
        help="the files containing smiles",
    )
    parser.add_argument(
        "--source",
        choices=["plain", "module"],
        help="indicates how to read the files. "
        "If 'plain' is used the input files should only contain SMILES (one on each row), "
        "if 'module' is used the SMILES are loaded from by python module"
        " (see documentation for details)",
        default="plain",
    )
    parser.add_argument(
        "--output",
        required=True,
        default="",
        help="the name of the output file or source tag",
    )
    parser.add_argument(
        "--target",
        choices=["hdf5", "mongo", "molbloom", "molbloom-inchi"],
        help="type of output",
        default="hdf5",
    )
    parser.add_argument("--host", help="the host of the Mongo database")
    parser.add_argument("--bloom_params", nargs=2, type=int, help="the parameters to the Bloom filter")
    return parser.parse_args()


def _convert_smiles(smiles_list: Iterable[str]) -> Iterable[str]:
    for smiles in smiles_list:
        try:
            yield Molecule(smiles=smiles, sanitize=True).inchi_key
        except MoleculeException:
            print(
                f"Failed to convert {smiles} to inchi key. Probably due to sanitation.",
                flush=True,
            )


def extract_plain_smiles(files: list[str]) -> Iterable[str]:
    """
    Extract SMILES from plain text files, one SMILES on each line.
    The SMILES are yielded to save memory.
    """
    for filename in files:
        print(f"Processing {filename}", flush=True)
        with open(filename) as fileobj:
            for line in fileobj:
                yield line.strip()


def extract_smiles_from_module(files: list[str]) -> Iterable[str]:
    """
    Extract SMILES by loading a custom module, containing
    the function ``extract_smiles``.

    The first element of the input argument is taken as the module name.
    The other elements are taken as input to the ``extract_smiles`` method

    The SMILES are yielded to save memory.
    """
    module_name = files.pop(0)
    module = importlib.import_module(module_name)
    if not files:
        for smiles in module.extract_smiles():  # type: ignore  # pylint: disable=R1737
            yield smiles
    else:
        for filename in files:
            print(f"Processing {filename}", flush=True)
            for smiles in module.extract_smiles(filename):  # type: ignore  # pylint: disable=R1737
                yield smiles


def make_hdf5_stock(inchi_keys: Iterable[str], filename: str) -> None:
    """
    Put all the inchi keys from the given iterable in a pandas
    dataframe and save it as an HDF5 file. Only unique inchi keys
    are stored.
    """
    data = pd.DataFrame.from_dict({"inchi_key": inchi_keys})
    data = data.drop_duplicates("inchi_key")
    data.to_hdf(filename, "table")
    print(f"Created HDF5 stock with {len(data)} unique compounds")


def main() -> None:
    """Entry-point for the smiles2stock tool"""
    args = _get_arguments()
    if args.source == "plain":
        smiles_gen = (smiles for smiles in extract_plain_smiles(args.files))
    else:
        smiles_gen = (smiles for smiles in extract_smiles_from_module(args.files))

    inchi_keys_gen = (inchi_key for inchi_key in _convert_smiles(smiles_gen))

    if args.target == "hdf5":
        make_hdf5_stock(inchi_keys_gen, args.output)


if __name__ == "__main__":
    main()
