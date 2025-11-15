import gzip
import hashlib
from pathlib import Path

from retrocast.exceptions import RetroCastException
from retrocast.utils.logging import logger


def get_content_hash(path: Path, block_size: int = 65536) -> str:
    """
    Computes the SHA256 hash of a file's uncompressed content,
    supporting both plain and gzipped files.
    """
    hasher = hashlib.sha256()

    open_func = gzip.open if path.suffix == ".gz" else open
    mode = "rb"

    with open_func(path, mode) as f:
        while block := f.read(block_size):
            hasher.update(block)

    return hasher.hexdigest()


def generate_file_hash(path: Path) -> str:
    """Computes the sha256 hash of a file's content."""
    try:
        with path.open("rb") as f:
            file_bytes = f.read()
            return hashlib.sha256(file_bytes).hexdigest()
    except OSError as e:
        logger.error(f"Could not read file for hashing: {path}")
        raise RetroCastException(f"File I/O error on {path}: {e}") from e


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
    return f"retrocasted-model-{full_hash[:8]}"


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
    return f"retrocasted-source-{hasher.hexdigest()}"
