from __future__ import annotations

import json

from retrocast.v2.cli.manifest import manifest_sidecar_path, write_manifest


def test_manifest_sidecar_path_handles_unknown_suffix(tmp_path) -> None:
    assert manifest_sidecar_path(tmp_path / "artifact.bin") == tmp_path / "artifact.manifest.json"


def test_write_manifest_keeps_absolute_paths_outside_root(tmp_path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    output = root / "out.txt"
    manifest_path = root / "manifest.json"
    outside.write_text("source", encoding="utf-8")
    output.parent.mkdir(parents=True)
    output.write_text("output", encoding="utf-8")

    write_manifest(
        manifest_path,
        action="test:v2",
        sources=[outside],
        outputs=[output],
        root_dir=root,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["source_files"][0]["path"] == str(outside.resolve())
    assert payload["output_files"][0]["path"] == "out.txt"
