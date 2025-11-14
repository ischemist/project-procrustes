"""Tests for the ursa.io module.

This module contains unit tests for all public I/O utilities, including
JSON (gzipped and uncompressed), CSV, and SMILES canonicalization.
"""

import csv
import json
import pathlib
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from retrocast.exceptions import UrsaException, UrsaIOException, UrsaSerializationError
from retrocast.io import (
    combine_evaluation_results,
    find_json_file_by_name,
    find_json_file_by_position,
    load_and_prepare_targets,
    load_json_gz,
    load_target_ids,
    load_targets_csv,
    load_targets_json,
    normalize_target_id_for_filename,
    save_json,
    save_json_gz,
)

VALID_TARGET_DATA = {"target_abc": "CCO", "target_xyz": "c1ccccc1"}


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Create a CSV file with two valid target rows."""
    file_path = tmp_path / "targets.csv"
    with file_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Structure ID", "SMILES"])
        writer.writerow(["target_abc", "CCO"])
        writer.writerow(["target_xyz", "c1ccccc1"])
    return file_path


@pytest.fixture
def valid_json_file(tmp_path: Path) -> Path:
    """Create an uncompressed JSON file with two valid targets."""
    file_path = tmp_path / "targets.json"
    file_path.write_text(json.dumps(VALID_TARGET_DATA))
    return file_path


@pytest.fixture
def valid_json_gz_file(tmp_path: Path) -> Path:
    """Create a gzipped JSON file with two valid targets."""
    file_path = tmp_path / "targets.json.gz"
    save_json_gz(VALID_TARGET_DATA, file_path)
    return file_path


def test_save_and_load_json_gz_roundtrip(tmp_path: Path) -> None:
    """Ensure gzipped JSON round-trip preserves nested structures."""
    data = {"key": "value", "nested": [1, 2, {"a": 3}]}
    file_path = tmp_path / "test.json.gz"
    save_json_gz(data, file_path)
    assert load_json_gz(file_path) == data


def test_save_and_load_uncompressed_json_roundtrip(tmp_path: Path) -> None:
    """Ensure uncompressed JSON round-trip preserves manifest data."""
    manifest = {"run_hash": "abc-123", "model_name": "test"}
    file_path = tmp_path / "manifest.json"
    save_json(manifest, file_path)
    assert json.loads(file_path.read_text()) == manifest


def test_save_json_gz_creates_directories(tmp_path: Path) -> None:
    """Verify that save_json_gz creates missing parent directories."""
    file_path = tmp_path / "processed" / "run_1" / "results.json.gz"
    save_json_gz({"status": "ok"}, file_path)
    assert file_path.exists()


def test_load_json_gz_raises_io_error_for_missing_file(tmp_path: Path) -> None:
    """Missing files raise UrsaIOException when loading gzipped JSON."""
    with pytest.raises(UrsaIOException):
        load_json_gz(tmp_path / "nope.json.gz")


def test_load_json_gz_raises_io_error_for_malformed_gzip(tmp_path: Path) -> None:
    """Malformed gzip content raises UrsaIOException."""
    file_path = tmp_path / "bad.json.gz"
    file_path.write_text("not gzipped")
    with pytest.raises(UrsaIOException):
        load_json_gz(file_path)


def test_save_json_gz_raises_serialization_error(tmp_path: Path) -> None:
    """Non-serializable objects raise UrsaSerializationError."""
    with pytest.raises(UrsaSerializationError):
        save_json_gz({"a_set": {1, 2, 3}}, tmp_path / "bad.json.gz")


def test_load_json_gz_raises_on_non_dict_content(tmp_path: Path) -> None:
    """Non-dict content raises UrsaIOException when loading gzipped JSON."""
    file_path = tmp_path / "list.json.gz"
    save_json_gz([1, 2, 3], file_path)  # type: ignore
    with pytest.raises(UrsaIOException):
        load_json_gz(file_path)


def test_load_targets_csv_happy_path(valid_csv_file: Path):
    """CSV loader returns expected dict for well-formed input."""
    assert load_targets_csv(valid_csv_file) == VALID_TARGET_DATA


def test_load_targets_csv_raises_on_missing_column(tmp_path: Path):
    """CSV loader raises when required columns are absent."""
    file_path = tmp_path / "bad_headers.csv"
    file_path.write_text("Structure ID,Wrong_Header\n-,-")
    with pytest.raises(UrsaIOException):
        load_targets_csv(file_path)


def test_load_targets_csv_empty_file(tmp_path: Path, caplog):
    """Empty CSV files return an empty dict and log a warning."""
    file_path = tmp_path / "empty.csv"
    file_path.touch()
    assert load_targets_csv(file_path) == {}
    assert "is empty or has no header" in caplog.text


def test_load_targets_csv_header_only(tmp_path: Path):
    """CSV files with only headers return an empty dict."""
    file_path = tmp_path / "header_only.csv"
    file_path.write_text("Structure ID,SMILES\n")
    assert load_targets_csv(file_path) == {}


def test_load_targets_json_happy_path(valid_json_file: Path):
    """JSON loader returns expected dict for well-formed input."""
    assert load_targets_json(valid_json_file) == VALID_TARGET_DATA


def test_load_targets_json_gz_happy_path(valid_json_gz_file: Path):
    """Gzipped JSON loader returns expected dict for well-formed input."""
    assert load_targets_json(valid_json_gz_file) == VALID_TARGET_DATA


def test_load_targets_json_raises_on_non_dict(tmp_path: Path):
    """Non-dict JSON content raises UrsaIOException."""
    file_path = tmp_path / "list.json"
    file_path.write_text("[1, 2, 3]")
    with pytest.raises(UrsaIOException):
        load_targets_json(file_path)


def test_prepare_targets_from_csv(valid_csv_file: Path):
    """Target preparation from CSV yields canonical SMILES."""
    targets = load_and_prepare_targets(valid_csv_file)
    assert targets["target_abc"].smiles == "CCO"
    assert targets["target_xyz"].smiles == "c1ccccc1"


def test_prepare_targets_from_json(valid_json_file: Path):
    """Target preparation from JSON yields expected target IDs."""
    targets = load_and_prepare_targets(valid_json_file)
    assert "target_abc" in targets


def test_prepare_targets_from_json_gz(valid_json_gz_file: Path):
    """Target preparation from gzipped JSON yields expected target IDs."""
    targets = load_and_prepare_targets(valid_json_gz_file)
    assert "target_xyz" in targets


def test_prepare_targets_raises_on_unsupported_format(tmp_path: Path):
    """Unsupported file extensions raise UrsaException."""
    file_path = tmp_path / "data.txt"
    file_path.touch()
    with pytest.raises(UrsaException):
        load_and_prepare_targets(file_path)


def test_prepare_targets_raises_on_invalid_smiles(tmp_path: Path):
    """Invalid SMILES strings raise UrsaException with a clear message."""
    file_path = tmp_path / "bad_smiles.csv"
    with file_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Structure ID", "SMILES"])
        writer.writerow(["good", "CCO"])
        writer.writerow(["bad", "invalid"])
    with pytest.raises(UrsaException):
        load_and_prepare_targets(file_path)


def test_prepare_targets_propagates_io_error(tmp_path: Path):
    """Non-existent files raise UrsaIOException."""
    with pytest.raises(UrsaIOException):
        load_and_prepare_targets(tmp_path / "non_existent_file.csv")


def test_save_json_gz_raises_io_error_on_write_failure(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Write failures during save_json_gz raise UrsaIOException."""
    file_path = tmp_path / "protected" / "data.json.gz"

    def mock_mkdir(*_, **__):
        raise OSError("Permission denied")

    monkeypatch.setattr(pathlib.Path, "mkdir", mock_mkdir)
    with pytest.raises(UrsaIOException):
        save_json_gz({"key": "value"}, file_path)


def test_save_json_raises_io_error_on_write_failure(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Write failures during save_json raise UrsaIOException."""
    file_path = tmp_path / "protected" / "data.json"

    def mock_mkdir(*_, **__):
        raise OSError("Disk is full")

    monkeypatch.setattr(pathlib.Path, "mkdir", mock_mkdir)
    with pytest.raises(UrsaIOException):
        save_json({"key": "value"}, file_path)


def test_load_targets_csv_raises_io_error_on_read_failure(monkeypatch: MonkeyPatch):
    """Read failures during CSV loading raise UrsaIOException."""

    def mock_open(*_, **__):
        raise OSError("Cannot read file")

    monkeypatch.setattr(Path, "open", mock_open)
    with pytest.raises(UrsaIOException):
        load_targets_csv(Path("/fake/path.csv"))


def test_load_targets_json_with_empty_object(tmp_path: Path, caplog):
    """Empty JSON objects return an empty dict and log a warning."""
    file_path = tmp_path / "empty.json"
    file_path.write_text("{}")
    assert load_targets_json(file_path) == {}
    assert "JSON file" in caplog.text and "is empty" in caplog.text


def test_load_targets_json_raises_on_malformed_json(tmp_path: Path):
    file_path = tmp_path / "malformed.json"
    file_path.write_text("{'key': 'invalid'}")
    with pytest.raises(UrsaIOException):
        load_targets_json(file_path)


def test_normalize_target_id_for_filename():
    """Test normalization of target IDs for filename use."""
    assert normalize_target_id_for_filename("A B") == "A_B"
    assert normalize_target_id_for_filename("A(B)C") == "A_B_C"
    assert normalize_target_id_for_filename("A__B") == "A_B"
    assert normalize_target_id_for_filename("USPTO-1/190") == "USPTO-1_190"
    assert normalize_target_id_for_filename("normal") == "normal"


@pytest.fixture
def valid_targets_csv(tmp_path: Path) -> Path:
    """Create a CSV file with target IDs for testing."""
    file_path = tmp_path / "targets.csv"
    with file_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Structure ID", "SMILES"])
        writer.writerow(["target_1", "CCO"])
        writer.writerow(["target_2", "c1ccccc1"])
    return file_path


def test_load_target_ids(valid_targets_csv: Path):
    """Test loading target IDs with positions."""
    result = load_target_ids(str(valid_targets_csv))
    assert result == [(1, "target_1"), (2, "target_2")]


def test_load_target_ids_raises_on_missing_file():
    """Missing CSV raises UrsaIOException."""
    with pytest.raises(UrsaIOException):
        load_target_ids("non_existent.csv")


def test_find_json_file_by_position(tmp_path: Path):
    """Test finding JSON file by position."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    file_path = eval_dir / "0001_test.json"
    file_path.write_text('{"data": "test"}')

    result = find_json_file_by_position(1, eval_dir)
    assert result == file_path

    # Non-existent
    assert find_json_file_by_position(2, eval_dir) is None


def test_find_json_file_by_name(tmp_path: Path):
    """Test finding JSON file by name."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    file_path = eval_dir / "target_1.json"
    file_path.write_text('{"data": "test"}')

    result = find_json_file_by_name("target_1", eval_dir, prefix="")
    assert result == file_path

    # With prefix
    file_path2 = eval_dir / "_target_2.json"
    file_path2.write_text('{"data": "test2"}')
    result2 = find_json_file_by_name("target_2", eval_dir, prefix="*_")
    assert result2 == file_path2

    # Non-existent
    assert find_json_file_by_name("missing", eval_dir, prefix="") is None


def test_combine_evaluation_results_askcos(tmp_path: Path, valid_targets_csv: Path):
    """Test combining results for ASKCOS naming convention."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Create files like 0001_target_1.json
    (eval_dir / "0001_target_1.json").write_text('{"result": "ok1"}')
    (eval_dir / "0002_target_2.json").write_text('{"result": "ok2"}')

    output_path = tmp_path / "results.json.gz"
    combine_evaluation_results(str(valid_targets_csv), str(eval_dir), str(output_path), "askcos")

    # Check output
    import gzip

    with gzip.open(output_path, "rt") as f:
        data = json.load(f)
    assert data == {"target_1": {"result": "ok1"}, "target_2": {"result": "ok2"}}


def test_combine_evaluation_results_chimera(tmp_path: Path, valid_targets_csv: Path):
    """Test combining results for Chimera naming convention."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Create files like target_1.json
    (eval_dir / "target_1.json").write_text('{"result": "ok1"}')
    (eval_dir / "target_2.json").write_text('{"result": "ok2"}')

    output_path = tmp_path / "results.json.gz"
    combine_evaluation_results(str(valid_targets_csv), str(eval_dir), str(output_path), "chimera")

    # Check output
    import gzip

    with gzip.open(output_path, "rt") as f:
        data = json.load(f)
    assert data == {"target_1": {"result": "ok1"}, "target_2": {"result": "ok2"}}


def test_combine_evaluation_results_missing_dir(tmp_path: Path, valid_targets_csv: Path):
    """Test that missing eval dir raises UrsaIOException."""
    eval_dir = tmp_path / "missing"
    output_path = tmp_path / "results.json.gz"
    with pytest.raises(UrsaIOException):
        combine_evaluation_results(str(valid_targets_csv), str(eval_dir), str(output_path), "askcos")


def test_combine_evaluation_results_unknown_convention(tmp_path: Path, valid_targets_csv: Path):
    """Test that unknown naming convention raises UrsaException."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    output_path = tmp_path / "results.json.gz"
    with pytest.raises(UrsaException):
        combine_evaluation_results(str(valid_targets_csv), str(eval_dir), str(output_path), "unknown")


def test_find_json_file_by_name_special_cases(tmp_path: Path):
    """Test finding JSON files with special characters in the name like (R)-Crizotinib."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    # Case 1: (R)-Crizotinib style
    crizotinib_file = eval_dir / "0001_Crizotinib.json"
    crizotinib_file.touch()
    result = find_json_file_by_name("(R)-Crizotinib", eval_dir, prefix="*_")
    assert result == crizotinib_file

    # Case 2: Normalization
    normalized_file = eval_dir / "0002_a_b_c.json"
    normalized_file.touch()
    result_norm = find_json_file_by_name("a(b)c", eval_dir, prefix="*_")
    assert result_norm == normalized_file


def test_combine_evaluation_results_askcos_fallback(tmp_path: Path, valid_targets_csv: Path):
    """Test that ASKCOS combination falls back from position- to name-based matching."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    # For target_1 (position 1), create a position-based match.
    (eval_dir / "0001_some_other_name.json").write_text('{"result": "pos_match"}')
    # For target_2 (position 2), do NOT create a 0002_* file. Create a name-based match.
    (eval_dir / "_target_2.json").write_text('{"result": "name_fallback"}')

    output_path = tmp_path / "results.json.gz"
    combine_evaluation_results(str(valid_targets_csv), str(eval_dir), str(output_path), "askcos")

    import gzip

    with gzip.open(output_path, "rt") as f:
        data = json.load(f)
    assert data["target_1"]["result"] == "pos_match"
    assert data["target_2"]["result"] == "name_fallback"


def test_combine_evaluation_results_with_missing_and_malformed_files(tmp_path: Path, caplog):
    """
    Test that result combination handles missing files and malformed JSON gracefully,
    and logs a final summary of all missing targets.
    """
    # Arrange: 3 targets, one good, one malformed, one missing
    targets_csv_path = tmp_path / "targets.csv"
    with targets_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Structure ID", "SMILES"])
        writer.writerow(["good_target", "C"])
        writer.writerow(["malformed_target", "CC"])
        writer.writerow(["missing_target", "CCC"])

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "good_target.json").write_text('{"result": "ok"}')
    (eval_dir / "malformed_target.json").write_text('{"result": "this is not valid json,}')

    output_path = tmp_path / "results.json.gz"

    # Act
    combine_evaluation_results(str(targets_csv_path), str(eval_dir), str(output_path), "chimera")

    # Assert
    import gzip

    with gzip.open(output_path, "rt") as f:
        data = json.load(f)

    assert "good_target" in data
    assert "malformed_target" not in data
    assert "missing_target" not in data
    assert len(data) == 1

    assert "missing files for 2 targets" in caplog.text.lower()
    assert "malformed_target" in caplog.text
    assert "missing_target" in caplog.text
