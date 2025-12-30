import logging
from enum import Enum

from rdkit import Chem, rdBase
from rdkit.Chem import rdinchi, rdMolDescriptors

from retrocast.exceptions import InvalidSmilesError, RetroCastException
from retrocast.typing import InchiKeyStr, SmilesStr

logger = logging.getLogger(__name__)

rdBase.DisableLog("rdApp.*")

# standard "no stereo" hash block for the second segment of an InChiKey, includes the standard flags (SA)
NO_STEREO_PLACEHOLDER = "UHFFFAOYSA"


class InchiKeyLevel(str, Enum):
    """
    Levels of InChI key specificity for chemical comparison.

    InChI keys have three blocks (25 hash chars, 2 hyphens, 27 chars total):
    - First 14 chars: Molecular connectivity (skeleton, hydrogens, charge)
    - Next 8 chars: Stereochemistry and isotopes
    - Last 3 chars: Standard/non-standard flag (S/N), version (A for v1), protonation (N)

    Example: BQJCRHHNABKAKU-KBQPJGBKSA-N
             └── 14 ────┘   └── 8 ─┘└─3┘
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


def _get_mol(smiles: str, func_name: str) -> Chem.Mol:
    """
    internal helper. parses smiles, handles None checks, sanitizes inputs.
    single point of failure for parsing logic.
    """
    if not isinstance(smiles, str) or not smiles:
        msg = f"SMILES input must be a non-empty string in {func_name}"
        logger.error(msg)
        raise InvalidSmilesError(msg)

    # rdkit handles exceptions poorly, usually just returns None or segfaults (rarely now)
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        # rare edge case where rdkit raises instead of returning None
        raise RetroCastException(f"RDKit raised error parsing '{smiles}': {e}") from e

    if mol is None:
        raise InvalidSmilesError(f"Invalid SMILES string: {smiles}")

    return mol


def canonicalize_smiles(smiles: str, remove_mapping: bool = False, ignore_stereo: bool = False) -> SmilesStr:
    """
    Converts a SMILES string to its canonical form using RDKit.

    Args:
        smiles: The input SMILES string.
        remove_mapping: If True, removes atom mapping numbers from the SMILES. Defaults to False.
        ignore_stereo: If True, strips stereochemistry information (dangerous - loses information).
            Defaults to False (stereochemistry is preserved).

    Returns:
        The canonical SMILES string.

    Raises:
        InvalidSmilesError: If the input SMILES is malformed or cannot be parsed by RDKit.
        RetroCastException: For any other unexpected errors during processing.
    """
    try:
        mol = _get_mol(smiles, "canonicalize_smiles")
        if remove_mapping:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

        # round trip ensures sanitization
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not ignore_stereo)
        return SmilesStr(canon)
    except (InvalidSmilesError, RetroCastException):
        raise
    except Exception as e:
        raise RetroCastException(f"Unexpected RDKit error canonicalizing '{smiles}': {e}") from e


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
        'BTANRVKWQNVYAZ-BYPYZUCNSA-N'
        >>> get_inchi_key("C[C@H](O)CC", level=InchiKeyLevel.NO_STEREO)  # No stereo
        'BTANRVKWQNVYAZ-UHFFFAOYSA-N'
        >>> get_inchi_key("C[C@H](O)CC", level=InchiKeyLevel.CONNECTIVITY)
        'BTANRVKWQNVYAZ'
    """
    try:
        mol = _get_mol(smiles, "get_inchi_key")

        if level == InchiKeyLevel.NO_STEREO:
            # Use -SNon to generate InChI without stereochemistry
            # MolToInchi returns (inchi, ret_code, message, log, aux_info)
            inchi, ret_code, _, _, _ = rdinchi.MolToInchi(mol, options="-SNon")
            if ret_code != 0:
                raise RetroCastException(f"rdkit failed to generate inchi (code {ret_code})")
            key = rdinchi.InchiToInchiKey(inchi)
        else:
            key = Chem.MolToInchiKey(mol)

        if not key:
            raise RetroCastException(f"Empty InchiKey generated for '{smiles}'")

        if level == InchiKeyLevel.CONNECTIVITY:
            return InchiKeyStr(key.split("-")[0])

        return InchiKeyStr(key)

    except (InvalidSmilesError, RetroCastException):
        raise
    except Exception as e:
        raise RetroCastException(f"unexpected error generating inchikey: {e}") from e


def reduce_inchikey(inchikey: str, level: InchiKeyLevel) -> InchiKeyStr:
    """
    Reduces an existing InChI key to a lower specificity level (destructive).

    Raises error if attempting to restore lost information.

    Use this when you have a full InChI key and need to reduce it for comparison.
    For generating keys directly at a specific level, use `get_inchi_key(smiles, level=...)`.

    Args:
        inchikey: A standard 27-character InChI key.
        level: Target level of specificity:
            - FULL: Returns the key unchanged
            - NO_STEREO: Replaces the stereo block with the standard no-stereo
              placeholder "UHFFFAOYSA", returning a valid 27-char InChI key.
              This matches the output of `get_inchi_key(smiles, level=InchiKeyLevel.NO_STEREO)`.
            - CONNECTIVITY: Returns first 14 chars only (molecular skeleton)

    Returns:
        The normalized InChI key at the specified level.

    Example:
        >>> reduce_inchikey("BQJCRHHNABKAKU-KBQPJGBKSA-N", InchiKeyLevel.NO_STEREO)
        'BQJCRHHNABKAKU-UHFFFAOYSA-N'
        >>> reduce_inchikey("BQJCRHHNABKAKU-KBQPJGBKSA-N", InchiKeyLevel.CONNECTIVITY)
        'BQJCRHHNABKAKU'
    """
    # Standard no-stereo placeholder used by InChI when stereo is not encoded
    parts = inchikey.split("-")

    # basic validation: 14 chars (1 part) or 27 chars (3 parts)
    if len(parts) not in (1, 3):
        raise ValueError(f"malformed inchikey structure: {inchikey}")

    # validate connectivity block is exactly 14 chars
    if len(parts[0]) != 14:
        raise ValueError(f"invalid connectivity block length (expected 14, got {len(parts[0])}): {inchikey}")

    current_is_partial = len(parts) == 1
    target_is_full = level in (InchiKeyLevel.FULL, InchiKeyLevel.NO_STEREO)

    # prevent upscaling (14 -> 25)
    if current_is_partial and target_is_full:
        raise RetroCastException(
            f"cannot upscale connectivity key '{inchikey}' to level '{level}'. information has been lost."
        )

    if level == InchiKeyLevel.FULL:
        # note: it is chemically impossible to detect if a 27-char key is "no_stereo" or just "a molecule with no stereo centers" just by looking at the string. so NO_STEREO -> FULL is allowed to pass through as an identity operation, which is pragmatically fine.
        return InchiKeyStr(inchikey)

    if level == InchiKeyLevel.CONNECTIVITY:
        return InchiKeyStr(parts[0])

    if level == InchiKeyLevel.NO_STEREO:
        # we already checked we have 3 parts above
        return InchiKeyStr(f"{parts[0]}-{NO_STEREO_PLACEHOLDER}-{parts[2]}")

    raise ValueError(f"unknown inchikey level: {level}")


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
