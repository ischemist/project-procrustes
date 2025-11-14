import hashlib
from pathlib import Path

from retrocast.exceptions import RetroCastException
from retrocast.typing import SmilesStr
from retrocast.utils.logging import logger


def generate_file_hash(path: Path) -> str:
    """Computes the sha256 hash of a file's content."""
    try:
        with path.open("rb") as f:
            file_bytes = f.read()
            return hashlib.sha256(file_bytes).hexdigest()
    except OSError as e:
        logger.error(f"Could not read file for hashing: {path}")
        raise RetroCastException(f"File I/O error on {path}: {e}") from e


def generate_molecule_hash(smiles: SmilesStr) -> str:
    """
    Generates a deterministic, content-based hash for a canonical SMILES string.

    Args:
        smiles: The canonical SMILES string.

    Returns:
        A 'sha256:' prefixed hex digest of the SMILES string.
    """
    # we encode to bytes before hashing
    smiles_bytes = smiles.encode("utf-8")
    hasher = hashlib.sha256(smiles_bytes)
    return f"sha256-{hasher.hexdigest()}"


def generate_model_hash(model_name: str) -> str:
    """
    Generates a short, deterministic, and stable hash for a model's name.

    This provides a consistent identifier for organizing outputs. We truncate
    a sha256 hash to 12 characters, which is sufficient to avoid collisions
    for any practical number of models.

    Returns:
        A 'retrocast-model-' prefixed, 12-character hex digest.
    """
    name_bytes = model_name.encode("utf-8")
    full_hash = hashlib.sha256(name_bytes).hexdigest()
    return f"retrocast-model-{full_hash[:8]}"


def generate_source_hash(model_name: str, file_hashes: list[str]) -> str:
    """
    Generates a full, deterministic hash for a specific run based on the
    model's name and the exact content of all its output files.

    This is used for cryptographic proof of what data was processed, and is
    stored in the manifest, NOT in the filename.

    Args:
        model_name: The name of the model being processed.
        file_hashes: A sorted list of the sha256 hashes of all input files.

    Returns:
        A 'retrocast-source-' prefixed full sha256 hex digest.
    """
    sorted_hashes = sorted(file_hashes)
    run_signature = model_name + "".join(sorted_hashes)
    run_bytes = run_signature.encode("utf-8")
    hasher = hashlib.sha256(run_bytes)
    return f"retrocast-source-{hasher.hexdigest()}"
