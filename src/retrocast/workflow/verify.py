"""
Core logic for manifest and data integrity verification using a two-phase audit.
"""

from pathlib import Path

from retrocast.io.provenance import calculate_file_hash
from retrocast.models.provenance import Manifest, VerificationReport


def _build_provenance_graph(start_path: Path, root_dir: Path, report: VerificationReport) -> dict[Path, Manifest]:
    """Recursively discover and load all manifests in the dependency graph."""
    graph: dict[Path, Manifest] = {}
    queue = [start_path]
    visited: set[Path] = set()

    while queue:
        manifest_path = queue.pop(0)
        if manifest_path in visited:
            continue
        visited.add(manifest_path)

        relative_path = manifest_path.relative_to(root_dir)
        if not manifest_path.exists():
            report.add("FAIL", relative_path, "Manifest file required for graph is missing.")
            continue

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = Manifest.model_validate_json(f.read())
            graph[relative_path] = manifest

            for source_file in manifest.source_files:
                parts = Path(source_file.path).parts
                is_primary = "1-benchmarks" in parts or "2-raw" in parts
                if not is_primary:
                    parent_manifest_path = root_dir / Path(source_file.path).parent / "manifest.json"
                    if parent_manifest_path not in visited:
                        queue.append(parent_manifest_path)
        except Exception as e:
            report.add("FAIL", relative_path, f"Failed to load or parse manifest: {e}")

    return graph


def _verify_logical_chain(graph: dict[Path, Manifest], report: VerificationReport) -> None:
    """Phase 1: Check for hash consistency between parent and child manifests."""
    report.add("INFO", report.manifest_path, "[Phase 1] Verifying logical consistency of the manifest chain...")

    for child_path, child_manifest in graph.items():
        report.add("INFO", child_path, f"Inspecting manifest for action '{child_manifest.action}'...")
        if not child_manifest.source_files:
            report.add("INFO", child_path, "-> No sources to verify, skipping.")
            continue

        for source_file in child_manifest.source_files:
            source_path = Path(source_file.path)
            parts = source_path.parts
            is_primary = "1-benchmarks" in parts or "2-raw" in parts

            if is_primary:
                report.add("PASS", source_path, "-> Source is a primary artifact, logical chain ends here.")
                continue

            parent_manifest_path = source_path.parent / "manifest.json"
            if parent_manifest_path not in graph:
                report.add(
                    "WARN",
                    source_path,
                    f"-> WARN: Parent manifest '{parent_manifest_path}' not found; cannot verify link.",
                )
                continue

            parent_manifest = graph[parent_manifest_path]
            parent_output_info = next(
                (out for out in parent_manifest.output_files if out.path == source_file.path), None
            )

            if not parent_output_info:
                report.add(
                    "FAIL",
                    source_path,
                    f"-> FAIL: Provenance broken. Not declared as output in parent manifest ('{parent_manifest.action}').",
                )
            elif parent_output_info.file_hash != source_file.file_hash:
                report.add(
                    "FAIL",
                    source_path,
                    f"-> FAIL: Provenance broken. Hash mismatch between manifests (Parent: {parent_output_info.file_hash[:8]}... vs. This: {source_file.file_hash[:8]}...).",
                )
            else:
                report.add(
                    "PASS", source_path, f"-> PASS: Link to parent manifest ('{parent_manifest.action}') is consistent."
                )


def _verify_physical_integrity(graph: dict[Path, Manifest], root_dir: Path, report: VerificationReport) -> None:
    """Phase 2: Check every file on disk against its defining manifest."""
    report.add("INFO", report.manifest_path, "[Phase 2] Verifying physical integrity of all files in the graph...")

    # Create a unique set of all relative file paths mentioned in the graph.
    all_files_to_check: set[Path] = set()
    for manifest in graph.values():
        for f in manifest.output_files:
            all_files_to_check.add(Path(f.path))
        for f in manifest.source_files:
            all_files_to_check.add(Path(f.path))

    for relative_path_str in sorted(list(all_files_to_check)):
        relative_path = Path(relative_path_str)
        # Find the manifest that DEFINES this file (i.e., where it's an output)
        defining_manifest_info = next(
            (m for m in graph.values() if any(out.path == relative_path for out in m.output_files)), None
        )

        # Or, if it's a primary source, it has no defining manifest in our graph
        is_primary = "1-benchmarks" in relative_path.parts or "2-raw" in relative_path.parts

        if defining_manifest_info:
            expected_hash = next(f.file_hash for f in defining_manifest_info.output_files if f.path == relative_path)
            context = f"defined by '{defining_manifest_info.action}'"
        elif is_primary:
            # For primary sources, we just check they exist. Hash is verified by downstream consumers.
            context = "primary source"
            expected_hash = None  # No single source of truth for its hash, only what consumers expect
        else:
            report.add(
                "WARN",
                relative_path,
                "-> WARN: File is a source but not a primary artifact and has no defining manifest in graph.",
            )
            continue

        absolute_path = root_dir / relative_path
        report.add("INFO", relative_path, f"Checking on-disk file ({context})...")

        if not absolute_path.exists():
            report.add("FAIL", relative_path, "-> FAIL: File is missing from disk.")
            continue

        if expected_hash:  # We only check hash if it's a generated artifact
            actual_hash = calculate_file_hash(absolute_path)
            if actual_hash != expected_hash:
                report.add(
                    "FAIL",
                    relative_path,
                    f"-> FAIL: Hash mismatch (Disk: {actual_hash[:8]}... vs. Manifest: {expected_hash[:8]}...).",
                )
            else:
                report.add("PASS", relative_path, "-> PASS: On-disk file hash matches its manifest definition.")
        else:  # For primary sources, just confirming existence is enough for this phase
            report.add("PASS", relative_path, "-> PASS: Primary source file exists on disk.")


def verify_manifest(manifest_path: Path, root_dir: Path, deep: bool = False) -> VerificationReport:
    """
    Verifies the integrity and lineage of an artifact via its manifest.
    """
    report = VerificationReport(manifest_path=manifest_path.relative_to(root_dir))

    if not deep:
        # Perform a simple, shallow verification if not deep
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = Manifest.model_validate_json(f.read())
            # A shallow check is just phase 2 on a single manifest
            _verify_physical_integrity({report.manifest_path: manifest}, root_dir, report)
        except Exception as e:
            report.add("FAIL", report.manifest_path, f"Failed to load manifest: {e}")
        return report

    # --- Deep Verification Starts Here ---

    # 1. Build the full dependency graph of all manifests.
    provenance_graph = _build_provenance_graph(manifest_path, root_dir, report)
    if not report.is_valid:
        report.add("FAIL", report.manifest_path, "Could not build provenance graph, aborting.")
        return report
    report.add(
        "PASS", report.manifest_path, f"Successfully built provenance graph with {len(provenance_graph)} manifests."
    )

    # 2. Phase 1: Verify the logical consistency of the entire graph.
    _verify_logical_chain(provenance_graph, report)
    if not report.is_valid:
        report.add("FAIL", report.manifest_path, "Logical chain verification failed, aborting physical check.")
        return report

    # 3. Phase 2: Verify the physical integrity of all files in the graph.
    _verify_physical_integrity(provenance_graph, root_dir, report)

    return report
