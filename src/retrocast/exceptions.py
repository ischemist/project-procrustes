class UrsaException(Exception):
    """Base exception for all errors raised by the ursa package."""

    pass


class InvalidSmilesError(UrsaException):
    """Raised when a SMILES string is malformed or cannot be processed."""

    pass


class SchemaLogicError(UrsaException, ValueError):
    """Raised when data violates the logical rules of a schema, beyond basic type validation."""

    pass


class AdapterLogicError(UrsaException):
    """Raised when an adapter fails to correctly fulfill its transformation contract."""

    pass


class UrsaIOException(UrsaException):
    """Raised for file system or I/O related errors during processing."""

    pass


class UrsaSerializationError(UrsaException):
    """Raised when data cannot be serialized to the desired format (e.g., JSON)."""

    pass


class TtlRetroSerializationError(UrsaSerializationError):
    """custom exception for errors during ttlretro route serialization."""

    pass


class SyntheseusSerializationError(UrsaSerializationError):
    """Custom exception for errors during syntheseus route serialization."""

    pass
