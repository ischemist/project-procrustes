class RetroCastException(Exception):
    """Base exception for all errors raised by the retrocast package."""

    pass


class InvalidSmilesError(RetroCastException):
    """Raised when a SMILES string is malformed or cannot be processed."""

    pass


class SchemaLogicError(RetroCastException, ValueError):
    """Raised when data violates the logical rules of a schema, beyond basic type validation."""

    pass


class BenchmarkValidationError(RetroCastException, ValueError):
    """Raised when benchmark data violates uniqueness or validation constraints."""

    pass


class AdapterLogicError(RetroCastException):
    """Raised when an adapter fails to correctly fulfill its transformation contract."""

    pass


class RetroCastIOError(RetroCastException):
    """Raised for file system or I/O related errors during processing."""

    pass


class RetroCastSerializationError(RetroCastException):
    """Raised when data cannot be serialized to the desired format (e.g., JSON)."""

    pass


class TtlRetroSerializationError(RetroCastSerializationError):
    """custom exception for errors during ttlretro route serialization."""

    pass


class SyntheseusSerializationError(RetroCastSerializationError):
    """Custom exception for errors during syntheseus route serialization."""

    pass
