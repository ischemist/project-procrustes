from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from retrocast._version import __version__


class FileInfo(BaseModel):
    """Metadata for a single file tracked by the manifest."""

    label: str | None = Field(default=None, description="Optional stable name for this tracked file.")
    path: str
    file_hash: str = Field(
        ...,
        description="SHA256 hash of the physical file",
        validation_alias=AliasChoices("file_hash", "sha256"),
        serialization_alias="sha256",
    )
    content_hash: str | None = Field(
        default=None, description="Semantic hash of the content (e.g. order-agnostic route hash)"
    )


class Manifest(BaseModel):
    """
    Provenance record for any data artifact produced by retrocast.
    """

    # manifests may carry extra fields from older scripts or newer producers; keep them round-trippable.
    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.1"
    retrocast_version: str = Field(default_factory=lambda: __version__)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # What was this run?
    action: str = Field(..., description="Name of the script or action (e.g., 'cast-paroutes')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Arguments/config used for this run")

    # Directives for retrocast consumption (e.g., which adapter to use, what filename to read)
    directives: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional directives for retrocast data consumption. "
            "Common directives: 'adapter', 'planner_version', 'raw_results_filename'"
        ),
    )
    release_name: str | None = Field(default=None, description="Optional stable release identifier.")

    # Inputs
    source_files: list[FileInfo] = Field(default_factory=list)

    # Outputs
    output_files: list[FileInfo] | dict[str, FileInfo] = Field(
        default_factory=list,
        description="Tracked outputs as either a flat list or a label-keyed dict. Prefer iter_output_files().",
    )

    # Optional stats (e.g., "n_targets_saved": 600)
    statistics: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)

    def iter_output_files(self) -> list[FileInfo]:
        """Return output files as a flat list regardless of storage shape."""
        if isinstance(self.output_files, dict):
            return list(self.output_files.values())
        return self.output_files


VerificationLevel = Literal["PASS", "FAIL", "WARN", "INFO"]
VerificationCategory = Literal["graph", "phase1", "phase2", "header", "context"]


class VerificationIssue(BaseModel):
    """A single issue found during verification."""

    level: VerificationLevel = Field(..., description="Severity of the issue.")
    path: Path = Field(..., description="The file or directory related to the issue.")
    message: str = Field(..., description="A human-readable description of the issue.")
    category: VerificationCategory | None = Field(default=None, description="Category of the verification issue.")


class VerificationReport(BaseModel):
    """The result of verifying a single manifest."""

    manifest_path: Path
    is_valid: bool = True
    issues: list[VerificationIssue] = Field(default_factory=list)

    def add(
        self, level: VerificationLevel, path: Path, message: str, category: VerificationCategory | None = None
    ) -> None:
        """Helper to add an issue and update validity."""
        self.issues.append(VerificationIssue(level=level, path=path, message=message, category=category))
        if level == "FAIL":
            self.is_valid = False
