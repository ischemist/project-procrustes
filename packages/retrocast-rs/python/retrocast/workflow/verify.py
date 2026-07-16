"""Python-facing provenance verification backed by the Rust engine."""

from __future__ import annotations

from pathlib import Path

from retrocast.models.provenance import VerificationReport


def verify_manifest(
    manifest_path: Path,
    root_dir: Path,
    deep: bool = False,
    output_only: bool = False,
    lenient: bool = True,
) -> VerificationReport:
    """Verify artifact integrity and lineage using the Rust provenance graph."""
    from retrocast import native

    return native.verify_manifest(
        str(manifest_path),
        str(root_dir),
        deep=deep,
        output_only=output_only,
        lenient=lenient,
    )
