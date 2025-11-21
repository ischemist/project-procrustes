"""
Core logic for manifest and data integrity verification.
"""

from pathlib import Path

from retrocast.io.blob import load_json_gz
from retrocast.io.provenance import calculate_file_hash
from retrocast.models.provenance import Manifest, VerificationReport


def verify_manifest(manifest_path: Path, root_dir: Path, deep: bool = False) -> VerificationReport:
    """
    Verifies the integrity and lineage of an artifact via its manifest.

    Args:
        manifest_path: Path to the manifest.json file to verify.
        root_dir: The root directory of the project (e.g., 'data/').
        deep: If True, also verifies the source files listed in the manifest.

    Returns:
        A VerificationReport object detailing the results.
    """
    report = VerificationReport(manifest_path=manifest_path)

    if not manifest_path.exists():
        report.add("FAIL", manifest_path, "Manifest file not found.")
        return report

    try:
        data = load_json_gz(manifest_path)
        manifest = Manifest.model_validate(data)
    except Exception as e:
        report.add("FAIL", manifest_path, f"Failed to load or parse manifest: {e}")
        return report

    # --- 1. Integrity Check (Output Files) ---
    for f_info in manifest.output_files:
        file_path = root_dir / f_info.path
        if not file_path.exists():
            report.add("FAIL", file_path, "Output file is missing.")
            continue

        actual_hash = calculate_file_hash(file_path)
        if actual_hash != f_info.file_hash:
            report.add(
                "FAIL",
                file_path,
                f"File hash mismatch. Expected: {f_info.file_hash[:12]}..., Found: {actual_hash[:12]}...",
            )
        else:
            report.add("PASS", file_path, "File integrity OK.")

    # --- 2. Lineage Check (Source Files) ---
    if deep:
        report.add("INFO", manifest_path, "Performing deep verification of sources...")
        if not manifest.source_files:
            report.add("WARN", manifest_path, "Deep verification requested, but manifest has no sources.")

        for f_info in manifest.source_files:
            file_path = root_dir / f_info.path
            if not file_path.exists():
                report.add("FAIL", file_path, "Source file is missing.")
                continue

            actual_hash = calculate_file_hash(file_path)
            if actual_hash != f_info.file_hash:
                report.add(
                    "FAIL",
                    file_path,
                    f"Source file hash mismatch. Manifest expected: {f_info.file_hash[:12]}..., Found: {actual_hash[:12]}...",
                )
            else:
                report.add("PASS", file_path, "Source file lineage OK.")

    if not report.issues:
        report.add("PASS", manifest_path, "All checks passed.")

    return report
