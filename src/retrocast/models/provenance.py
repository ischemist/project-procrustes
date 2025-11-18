from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from retrocast import __version__


class FileInfo(BaseModel):
    """Metadata for a single file tracked by the manifest."""

    path: str
    file_hash: str = Field(..., description="SHA256 hash of the physical file")
    content_hash: str | None = Field(
        default=None, description="Semantic hash of the content (e.g. order-agnostic route hash)"
    )


class Manifest(BaseModel):
    """
    Provenance record for any data artifact produced by retrocast.
    """

    schema_version: str = "1.0"
    retrocast_version: str = Field(default_factory=lambda: __version__)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # What was this run?
    action: str = Field(..., description="Name of the script or action (e.g., 'cast-paroutes')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Arguments/config used for this run")

    # Inputs
    source_files: list[FileInfo] = Field(default_factory=list)

    # Outputs
    output_files: list[FileInfo] = Field(default_factory=list)

    # Optional stats (e.g., "n_targets_saved": 600)
    statistics: dict[str, Any] = Field(default_factory=dict)
