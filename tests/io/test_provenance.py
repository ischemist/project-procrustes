from __future__ import annotations

import logging
from pathlib import Path

import pytest

from retrocast.io.provenance import ContentType, calculate_file_hash, create_manifest
from retrocast.models.analysis import AnalysisReport, MetricSummary


@pytest.mark.unit
def test_manifest_supports_keyed_labeled_outputs_and_stock_identity_hash(tmp_path: Path) -> None:
    output = tmp_path / "stock.csv.gz"
    output.write_text("SMILES,InChIKey\nC,key-c\n", encoding="utf-8")
    stock_a = {"key-c": "C", "key-o": "O"}
    stock_b = {"key-o": "CO", "key-c": "CC"}

    manifest_a = create_manifest(
        action="stock",
        sources=[],
        outputs=[("stock", output, stock_a, ContentType.STOCK)],
        root_dir=tmp_path,
        keyed_output_files=True,
    )
    manifest_b = create_manifest(
        action="stock",
        sources=[],
        outputs=[("stock", output, stock_b, "stock")],
        root_dir=tmp_path,
        keyed_output_files=True,
    )

    assert isinstance(manifest_a.output_files, dict)
    assert manifest_a.output_files["stock"].path == "stock.csv.gz"
    assert manifest_a.output_files["stock"].content_hash == manifest_b.output_files["stock"].content_hash


@pytest.mark.unit
def test_manifest_rejects_malformed_outputs_and_unlabeled_keyed_outputs(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="requires every output"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=[(output, {}, ContentType.UNKNOWN)],
            root_dir=tmp_path,
            keyed_output_files=True,
        )
    with pytest.raises(TypeError, match="label must be str"):
        create_manifest(action="bad", sources=[], outputs=[(1, output, {}, ContentType.UNKNOWN)], root_dir=tmp_path)
    with pytest.raises(TypeError, match="path must be Path"):
        create_manifest(action="bad", sources=[], outputs=[("not-path", {}, ContentType.UNKNOWN)], root_dir=tmp_path)
    with pytest.raises(TypeError, match="path must be Path"):
        create_manifest(
            action="bad", sources=[], outputs=[("label", "not-path", {}, ContentType.UNKNOWN)], root_dir=tmp_path
        )
    with pytest.raises(ValueError, match="3 or 4 items"):
        create_manifest(action="bad", sources=[], outputs=[(output,)], root_dir=tmp_path)  # type: ignore[list-item]


@pytest.mark.unit
def test_manifest_logs_unhashable_or_external_paths(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "missing.json"
    external = tmp_path.parent / "external.json"
    external.write_text("{}", encoding="utf-8")
    model = AnalysisReport(metrics={"m": MetricSummary(value=1.0, count=1)})

    caplog.set_level(logging.WARNING)
    manifest = create_manifest(
        action="external",
        sources=[],
        outputs=[(external, model, ContentType.BENCHMARK)],
        root_dir=tmp_path,
    )

    assert calculate_file_hash(missing) == "error-hashing-file"
    assert manifest.output_files[0].path == str(external.resolve())
    assert manifest.output_files[0].content_hash is not None
    assert "could not hash file" in caplog.text
    assert "is not inside root" in caplog.text


@pytest.mark.unit
def test_manifest_hashes_plain_jsonable_content(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")

    manifest = create_manifest(
        action="plain",
        sources=[],
        outputs=[(output, {"b": 2, "a": 1}, ContentType.BENCHMARK)],
        root_dir=tmp_path,
    )

    assert manifest.output_files[0].content_hash is not None
