from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    assert isinstance(manifest_b.output_files, dict)
    assert manifest_a.output_files["stock"].path == "stock.csv.gz"
    assert manifest_a.output_files["stock"].content_hash == manifest_b.output_files["stock"].content_hash


@pytest.mark.unit
def test_manifest_uses_explicit_labeled_output_content_hash(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")

    manifest = create_manifest(
        action="explicit",
        sources=[],
        outputs=[("artifact", output, {"ignored": True}, ContentType.UNKNOWN, "explicit-hash")],
        root_dir=tmp_path,
        keyed_output_files=True,
    )

    assert isinstance(manifest.output_files, dict)
    assert manifest.output_files["artifact"].content_hash == "explicit-hash"


@pytest.mark.unit
def test_manifest_rejects_malformed_outputs_and_unlabeled_keyed_outputs(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    invalid_label_output: list[Any] = [(1, output, {}, ContentType.UNKNOWN)]
    invalid_unlabeled_path_output: list[Any] = [("not-path", {}, ContentType.UNKNOWN)]
    invalid_labeled_path_output: list[Any] = [("label", "not-path", {}, ContentType.UNKNOWN)]
    invalid_content_hash_output: list[Any] = [("label", output, {}, ContentType.UNKNOWN, 42)]
    invalid_arity_output: list[Any] = [(output,)]

    with pytest.raises(ValueError, match="requires every output"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=[(output, {}, ContentType.UNKNOWN)],
            root_dir=tmp_path,
            keyed_output_files=True,
        )
    with pytest.raises(TypeError, match="label must be str"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=invalid_label_output,
            root_dir=tmp_path,
        )
    with pytest.raises(TypeError, match="path must be Path"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=invalid_unlabeled_path_output,
            root_dir=tmp_path,
        )
    with pytest.raises(TypeError, match="path must be Path"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=invalid_labeled_path_output,
            root_dir=tmp_path,
        )
    with pytest.raises(TypeError, match="content hash must be str"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=invalid_content_hash_output,
            root_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="3, 4, or 5 items"):
        create_manifest(
            action="bad",
            sources=[],
            outputs=invalid_arity_output,
            root_dir=tmp_path,
        )


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
    output_info = manifest.iter_output_files()[0]
    assert output_info.path == str(external.resolve())
    assert output_info.content_hash is not None
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

    assert manifest.iter_output_files()[0].content_hash is not None
