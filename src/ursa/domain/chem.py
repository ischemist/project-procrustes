from rdkit import Chem, rdBase

from ursa.exceptions import InvalidSmilesError, UrsaException
from ursa.typing import InchiKeyStr, SmilesStr
from ursa.utils.logging import logger

rdBase.DisableLog("rdApp.error")


def canonicalize_smiles(smiles: str, remove_mapping: bool = False) -> SmilesStr:
    """
    Converts a SMILES string to its canonical form using RDKit.

    Args:
        smiles: The input SMILES string.

    Returns:
        The canonical SMILES string.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        UrsaException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # this is rdkit's sad, C-style way of saying "parse failed"
            logger.warning(f"RDKit failed to parse SMILES: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")
        if remove_mapping:
            for atom in mol.GetAtoms():  # type: ignore
                atom.SetAtomMapNum(0)

        # we do a round trip to sanitize and be EXTRA sure.
        # some things parse but don't write. kekw.
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return SmilesStr(canonical_smiles)

    except InvalidSmilesError:
        # This is our own specific, expected exception. Let it pass through untouched.
        raise

    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred for SMILES '{smiles}': {e}")
        # wrap the unknown error so the rest of the app doesn't need to know about rdkit specifics
        raise UrsaException(f"An unexpected error occurred during SMILES processing: {e}") from e


def get_inchi_key(smiles: str) -> str:
    """
    Generates a standard InChIKey from a SMILES string.

    Args:
        smiles: The input SMILES string.

    Returns:
        The standard InChIKey string.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        UrsaException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES for InChIKey generation is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"RDKit failed to parse SMILES for InChIKey generation: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

        # MolToInchiKey is the canonical RDKit function for this.
        inchi_key = Chem.MolToInchiKey(mol)  # type: ignore
        if not inchi_key:
            # This is a very rare edge case but good to guard against.
            msg = f"RDKit produced an empty InChIKey for SMILES: '{smiles}'"
            logger.error(msg)
            raise UrsaException(msg)
        return InchiKeyStr(inchi_key)

    except InvalidSmilesError:
        # Re-raise our own specific, expected exception. Let it pass through untouched.
        raise
    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred during InChIKey generation for SMILES '{smiles}': {e}")
        # Wrap the unknown error so the rest of the app doesn't need to know about rdkit specifics
        raise UrsaException(f"An unexpected error occurred during InChIKey generation: {e}") from e
