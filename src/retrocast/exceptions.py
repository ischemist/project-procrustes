from __future__ import annotations

from typing import Any


class RetroCastException(Exception):
    """Base exception for boundary-facing errors raised by retrocast."""

    default_code = "retrocast.error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        self.code = code or self.default_code
        self.message = message
        self.context = context or {}
        self.retryable = retryable
        Exception.__init__(self, message)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the stable boundary shape used by logs, manifests, and tests."""
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "retryable": self.retryable,
        }


class InputError(RetroCastException, ValueError):
    """Raised when external input is invalid at a package, cli, or workflow boundary."""

    default_code = "input.invalid"


class ChemError(RetroCastException):
    """Raised for chemistry parsing, normalization, or identity failures."""

    default_code = "chem.error"


class DataIOError(RetroCastException):
    """Raised for expected artifact read, write, decode, or format failures."""

    default_code = "io.error"


class AdapterError(RetroCastException):
    """Raised for expected adapter resolution, schema, or route contract failures."""

    default_code = "adapter.error"


class WorkflowError(RetroCastException):
    """Raised when a workflow cannot complete an expected operation."""

    default_code = "workflow.error"


class BenchmarkCollectionError(WorkflowError):
    """Raised when benchmark collection cannot classify or place canonical routes."""

    default_code = "collection.error"


class PredictionCollectionError(BenchmarkCollectionError):
    """Deprecated alias for benchmark collection failures."""


class DatasetError(RetroCastException):
    """Raised for expected hosted dataset resolution, download, or verification failures."""

    default_code = "dataset.error"


class DatasetResolutionError(DatasetError):
    """Raised when a hosted dataset release or artifact cannot be resolved."""

    default_code = "dataset.resolution_failed"


class DatasetDownloadError(DatasetError):
    """Raised when a hosted dataset file cannot be downloaded."""

    default_code = "dataset.download_failed"


class DatasetVerificationError(DatasetError):
    """Raised when a downloaded dataset file fails integrity verification."""

    default_code = "dataset.verification_failed"


class TrainingReleaseError(WorkflowError):
    """Raised when training release construction violates a release contract."""

    default_code = "workflow.training_release_error"


class BenchmarkError(RetroCastException, ValueError):
    """Raised when benchmark construction or validation fails."""

    default_code = "benchmark.error"


class SecurityError(InputError):
    """Raised when security validation fails, such as path traversal or unsafe filenames."""

    default_code = "security.path_invalid"


class InvalidSmilesError(ChemError, ValueError):
    """Raised when a smiles string is malformed or cannot be processed."""

    default_code = "chem.invalid_smiles"


class ChemRuntimeError(ChemError):
    """Raised when rdkit or a chemistry backend fails unexpectedly."""

    default_code = "chem.runtime_error"


class InvalidInchiKeyError(ChemError, ValueError):
    """Raised when an inchikey is malformed or cannot be transformed as requested."""

    default_code = "chem.invalid_inchikey"


class SchemaLogicError(InputError):
    """Raised when data violates logical schema rules beyond basic type validation."""

    default_code = "schema.logic_error"


class BenchmarkValidationError(BenchmarkError):
    """Raised when benchmark data violates uniqueness or validation constraints."""

    default_code = "benchmark.validation_failed"


class AdapterLogicError(AdapterError):
    """Raised when an adapter cannot fulfill its route transformation contract."""

    default_code = "adapter.route_transform_failed"


class AdapterResolutionError(AdapterError):
    """Raised when no adapter can be resolved from the resolution hierarchy."""

    default_code = "adapter.resolution_failed"


class AdapterSchemaError(AdapterError):
    """Raised when raw adapter input fails expected schema validation."""

    default_code = "adapter.schema_invalid"


class UnsupportedAdapterFeatureError(AdapterError, NotImplementedError):
    """Raised when an adapter receives a valid request for an unsupported feature."""

    default_code = "adapter.unsupported_feature"


class RetroCastIOError(DataIOError):
    """Raised for file system or I/O related errors during processing."""

    default_code = "io.error"


class ArtifactNotFoundError(RetroCastIOError):
    """Raised when a required artifact is missing."""

    default_code = "io.not_found"


class ArtifactFormatError(RetroCastIOError, ValueError):
    """Raised when an artifact has an unsupported or invalid format."""

    default_code = "io.invalid_artifact_shape"


class ArtifactDecodeError(RetroCastIOError):
    """Raised when an artifact cannot be decoded or parsed."""

    default_code = "io.decode_failed"


class ArtifactWriteError(RetroCastIOError):
    """Raised when an artifact cannot be written."""

    default_code = "io.write_failed"


class ConfigurationError(InputError):
    """Raised when a configuration value is invalid or malformed."""

    default_code = "config.invalid_value"


class RetroCastSerializationError(RetroCastException):
    """Raised when data cannot be serialized to the desired format."""

    default_code = "serialization.failed"


class TtlRetroSerializationError(RetroCastSerializationError):
    """Raised for errors during ttlretro route serialization."""

    default_code = "serialization.ttlretro_failed"


class SyntheseusSerializationError(RetroCastSerializationError):
    """Raised for errors during syntheseus route serialization."""

    default_code = "serialization.syntheseus_failed"
