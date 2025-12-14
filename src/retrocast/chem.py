import logging
from enum import Enum

from rdkit import Chem, rdBase
from rdkit.Chem import rdinchi, rdMolDescriptors

from retrocast.exceptions import InvalidSmilesError, RetroCastException
from retrocast.typing import InchiKeyStr, SmilesStr

logger = logging.getLogger(__name__)

rdBase.DisableLog("rdApp.*")


class InchiKeyLevel(str, Enum):
    """
    Levels of InChI key specificity for chemical comparison.

    InChI keys have three blocks (27 chars total):
    - First 14 chars: Molecular connectivity (skeleton, hydrogens, charge)
    - Next 8 chars: Stereochemistry and isotopes
    - Last 5 chars: Standard/non-standard flag, version, protonation

    Example: BQJCRHHNABKAKU-KBQPJGBKSA-N
             └── 14 ────┘ └─ 8 ─┘ └5┘
    """

    # Full 27-char InChI key with all stereochemistry (default)
    FULL = "full"

    # Full 27-char InChI key generated WITHOUT stereochemistry info.
    # Uses -SNon option during InChI generation, producing a proper standard InChI.
    # The stereo block will be "UHFFFAOY" (all F's = no stereo info).
    NO_STEREO = "no_stereo"

    # First 14 characters only (connectivity layer).
    # Useful for pure structural identity regardless of stereo, isotopes, or protonation.
    # Warning: loses protonation information - use with care.
    CONNECTIVITY = "connectivity"


def canonicalize_smiles(smiles: str, remove_mapping: bool = False, isomeric: bool = True) -> SmilesStr:
    """
    Converts a SMILES string to its canonical form using RDKit.

    Args:
        smiles: The input SMILES string.

    Returns:
        The canonical SMILES string.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # this is rdkit's sad, C-style way of saying "parse failed"
            logger.debug(f"RDKit failed to parse SMILES: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")
        if remove_mapping:
            for atom in mol.GetAtoms():  # type: ignore
                atom.SetAtomMapNum(0)

        # we do a round trip to sanitize and be EXTRA sure.
        # some things parse but don't write. kekw.
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
        return SmilesStr(canonical_smiles)

    except InvalidSmilesError:
        # This is our own specific, expected exception. Let it pass through untouched.
        raise

    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred for SMILES '{smiles}': {e}")
        # wrap the unknown error so the rest of the app doesn't need to know about rdkit specifics
        raise RetroCastException(f"An unexpected error occurred during SMILES processing: {e}") from e


def get_inchi_key(smiles: str, level: InchiKeyLevel = InchiKeyLevel.FULL) -> InchiKeyStr:
    """
    Generates an InChIKey from a SMILES string with configurable specificity.

    Args:
        smiles: The input SMILES string.
        level: The level of specificity for the InChI key:
            - FULL: Standard 27-char InChI key with all stereo info (default)
            - NO_STEREO: Standard 27-char InChI key generated without stereochemistry.
              Uses the -SNon option during InChI generation, which produces a valid
              standard InChI. The stereo block will show "UHFFFAOY" indicating no
              stereo information was encoded.
            - CONNECTIVITY: First 14 characters only (molecular connectivity).
              Loses protonation info - use with care.

    Returns:
        The InChIKey string at the requested level of specificity.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.

    Examples:
        >>> get_inchi_key("C[C@H](O)CC")  # Full key with stereo
        'BQJCRHHNABKAKU-KBQPJGBKSA-N'
        >>> get_inchi_key("C[C@H](O)CC", level=InchiKeyLevel.NO_STEREO)  # No stereo
        'BQJCRHHNABKAKU-UHFFFAOYSA-N'
        >>> get_inchi_key("C[C@H](O)CC", level=InchiKeyLevel.CONNECTIVITY)
        'BQJCRHHNABKAKU'
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES for InChIKey generation is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"RDKit failed to parse SMILES for InChIKey generation: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

        # Generate InChI with appropriate options based on level
        if level == InchiKeyLevel.NO_STEREO:
            # Use -SNon to generate InChI without stereochemistry
            # MolToInchi returns (inchi, ret_code, message, log, aux_info)
            result = rdinchi.MolToInchi(mol, options="-SNon")
            inchi, ret_code = result[0], result[1]
            if ret_code != 0 or not inchi:
                msg = f"RDKit failed to generate InChI for SMILES: '{smiles}'"
                logger.error(msg)
                raise RetroCastException(msg)
            inchi_key = rdinchi.InchiToInchiKey(inchi)
        else:
            # Standard InChI generation (includes all stereo info)
            inchi_key = Chem.MolToInchiKey(mol)  # type: ignore

        if not inchi_key:
            msg = f"RDKit produced an empty InChIKey for SMILES: '{smiles}'"
            logger.error(msg)
            raise RetroCastException(msg)

        # For CONNECTIVITY level, return only the first 14 characters
        if level == InchiKeyLevel.CONNECTIVITY:
            inchi_key = inchi_key.split("-")[0]

        return InchiKeyStr(inchi_key)

    except InvalidSmilesError:
        raise
    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred during InChIKey generation for SMILES '{smiles}': {e}")
        raise RetroCastException(f"An unexpected error occurred during InChIKey generation: {e}") from e


def normalize_inchikey(inchikey: str, level: InchiKeyLevel) -> str:
    """
    Normalizes an existing InChI key to the specified level.

    Use this when you have a full InChI key and need to reduce it for comparison.
    For generating keys directly at a specific level, use `get_inchi_key(smiles, level=...)`.

    Args:
        inchikey: A standard 27-character InChI key.
        level: Target level of specificity:
            - FULL: Returns the key unchanged
            - NO_STEREO: Drops the stereo block (middle 8 chars), returns 20-char key
            - CONNECTIVITY: Returns first 14 chars only (molecular skeleton)

    Returns:
        The normalized InChI key at the specified level.

    Example:
        >>> normalize_inchikey("BQJCRHHNABKAKU-KBQPJGBKSA-N", InchiKeyLevel.NO_STEREO)
        'BQJCRHHNABKAKU-N'
        >>> normalize_inchikey("BQJCRHHNABKAKU-KBQPJGBKSA-N", InchiKeyLevel.CONNECTIVITY)
        'BQJCRHHNABKAKU'
    """
    if level == InchiKeyLevel.FULL:
        return inchikey
    elif level == InchiKeyLevel.NO_STEREO:
        parts = inchikey.split("-")
        if len(parts) == 3:
            return f"{parts[0]}-{parts[2]}"
        return inchikey
    elif level == InchiKeyLevel.CONNECTIVITY:
        return inchikey.split("-")[0]
    else:
        raise ValueError(f"Unknown InchiKeyLevel: {level}")


def get_heavy_atom_count(smiles: str) -> int:
    """
    Returns the number of heavy (non-hydrogen) atoms in a molecule.

    Args:
        smiles: The input SMILES string.

    Returns:
        The count of heavy atoms.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES for HAC calculation is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"RDKit failed to parse SMILES for HAC calculation: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

        return mol.GetNumAtoms()

    except InvalidSmilesError:
        raise
    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred during HAC calculation for SMILES '{smiles}': {e}")
        raise RetroCastException(f"An unexpected error occurred during HAC calculation: {e}") from e


def get_molecular_weight(smiles: str) -> float:
    """
    Returns the exact molecular weight of a molecule.

    Args:
        smiles: The input SMILES string.

    Returns:
        The exact molecular weight in Daltons.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES for MW calculation is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"RDKit failed to parse SMILES for MW calculation: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

        return rdMolDescriptors.CalcExactMolWt(mol)

    except InvalidSmilesError:
        raise
    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred during MW calculation for SMILES '{smiles}': {e}")
        raise RetroCastException(f"An unexpected error occurred during MW calculation: {e}") from e


def get_chiral_center_count(smiles: str) -> int:
    """
    Returns the number of chiral centers in a molecule.

    Args:
        smiles: The input SMILES string.

    Returns:
        The count of chiral centers.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.error("Provided SMILES for chiral center count is not a valid string or is empty.")
        raise InvalidSmilesError("SMILES input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"RDKit failed to parse SMILES for chiral center count: '{smiles}'")
            raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

        chiral_centers = Chem.FindMolChiralCenters(mol)
        return len(chiral_centers)

    except InvalidSmilesError:
        raise
    except Exception as e:
        logger.error(f"An unexpected RDKit error occurred during chiral center count for SMILES '{smiles}': {e}")
        raise RetroCastException(f"An unexpected error occurred during chiral center count: {e}") from e
