from enum import StrEnum
from typing import Any

from retrocast.exceptions import ChemRuntimeError, InvalidInchiKeyError, InvalidSmilesError
from retrocast.typing import InChIKeyStr, SmilesStr

try:
    from retrocast import _native
except ImportError:  # pragma: no cover - source checkout before `maturin develop`
    _native: Any | None = None


NO_STEREO_PLACEHOLDER = "UHFFFAOYSA"


class InChIKeyLevel(StrEnum):
    """The chemical layers retained in an InChIKey."""

    FULL = "full"
    NO_STEREO = "no_stereo"
    CONNECTIVITY = "connectivity"


def canonicalize_smiles(smiles: str, remove_mapping: bool = False, ignore_stereo: bool = False) -> SmilesStr:
    """Return RDKit's canonical SMILES, optionally removing mapping or stereo."""
    _validate_smiles(smiles, "canonicalize_smiles")
    try:
        return SmilesStr(_engine().canonicalize_smiles(smiles, remove_mapping, ignore_stereo))
    except ValueError as exc:
        raise _invalid_smiles(smiles, "canonicalize_smiles") from exc
    except RuntimeError as exc:
        raise ChemRuntimeError(
            f"Unexpected RDKit error canonicalizing '{smiles}': {exc}",
            code="chem.rdkit_canonicalize_failed",
            context={"operation": "canonicalize_smiles", "smiles": smiles},
        ) from exc


def get_inchi_key(smiles: str, level: InChIKeyLevel = InChIKeyLevel.FULL) -> InChIKeyStr:
    """Return an RDKit InChIKey at the requested identity level."""
    _validate_smiles(smiles, "get_inchi_key")
    resolved_level = _validate_level(level)
    try:
        return InChIKeyStr(_engine().get_inchi_key(smiles, resolved_level.value))
    except ValueError as exc:
        raise _invalid_smiles(smiles, "get_inchi_key") from exc
    except RuntimeError as exc:
        raise ChemRuntimeError(
            f"unexpected error generating inchikey: {exc}",
            code="chem.inchikey_generation_failed",
            context={"smiles": smiles, "level": resolved_level.value},
        ) from exc


def reduce_inchikey(inchikey: str, level: InChIKeyLevel) -> InChIKeyStr:
    """Destructively reduce an existing InChIKey to a coarser identity level."""
    resolved_level = _validate_level(level)
    try:
        return InChIKeyStr(_engine().reduce_inchi_key(inchikey, resolved_level.value))
    except ValueError as exc:
        code = "chem.inchikey_upscale" if "cannot upscale" in str(exc) else "chem.invalid_inchikey"
        raise InvalidInchiKeyError(
            str(exc),
            code=code,
            context={"inchikey": inchikey, "target_level": resolved_level.value},
        ) from exc


def get_heavy_atom_count(smiles: str) -> int:
    """Return the number of non-hydrogen atoms in a molecule."""
    return int(_descriptors(smiles, "get_heavy_atom_count", "HAC calculation")[0])


def get_molecular_weight(smiles: str) -> float:
    """Return the exact molecular weight in daltons."""
    return float(_descriptors(smiles, "get_molecular_weight", "MW calculation")[1])


def get_chiral_center_count(smiles: str) -> int:
    """Return the number of assigned chiral centers in a molecule."""
    return int(_descriptors(smiles, "get_chiral_center_count", "chiral center count")[2])


def _descriptors(smiles: str, operation: str, label: str) -> tuple[int, float, int]:
    _validate_smiles(smiles, operation)
    try:
        return _engine().molecular_descriptors(smiles)
    except ValueError as exc:
        raise _invalid_smiles(smiles, operation) from exc
    except RuntimeError as exc:
        raise ChemRuntimeError(
            f"An unexpected error occurred during {label}: {exc}",
            code="chem.descriptor_failed",
            context={"operation": operation, "smiles": smiles},
        ) from exc


def _engine() -> Any:
    if _native is None:
        raise ChemRuntimeError(
            "RetroCast's native chemistry engine is unavailable; install a platform wheel or run `maturin develop`.",
            code="chem.native_unavailable",
        )
    return _native


def _validate_smiles(smiles: object, operation: str) -> None:
    if not isinstance(smiles, str) or not smiles:
        raise InvalidSmilesError(
            f"SMILES input must be a non-empty string in {operation}",
            code="chem.invalid_smiles",
            context={"operation": operation, "value": smiles},
        )


def _invalid_smiles(smiles: str, operation: str) -> InvalidSmilesError:
    return InvalidSmilesError(
        f"Invalid SMILES string: {smiles}",
        code="chem.invalid_smiles",
        context={"operation": operation, "smiles": smiles},
    )


def _validate_level(level: InChIKeyLevel) -> InChIKeyLevel:
    if not isinstance(level, InChIKeyLevel):
        raise ValueError(f"unknown inchikey level: {level}")
    return level
