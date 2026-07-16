import ast
import gzip
import json
from pathlib import Path

import pytest

from retrocast.cli.errors import format_cli_error
from retrocast.exceptions import ArtifactDecodeError, ArtifactFormatError, RetroCastException
from retrocast.io import load_json_gz
from retrocast.io.data import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "retrocast"


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_retrocast_exception_serializes_stable_contract():
    error = RetroCastException(
        "boom",
        code="test.failure",
        context={"target_id": "t1"},
        retryable=True,
    )

    assert error.to_dict() == {
        "code": "test.failure",
        "message": "boom",
        "context": {"target_id": "t1"},
        "retryable": True,
    }


def test_cli_error_formatter_includes_code_and_context():
    error = RetroCastException(
        "bad input",
        code="security.path_invalid",
        context={"path": "../x"},
    )

    rendered = format_cli_error(error)

    assert "bad input" in rendered
    assert "[security.path_invalid]" in rendered
    assert "path=../x" in rendered


def test_cli_error_formatter_flattens_multiline_context():
    error = RetroCastException(
        "bad input",
        code="schema.invalid",
        context={"detail": "line one\nline two\rline three"},
    )

    rendered = format_cli_error(error)

    assert "detail=line one line two line three" in rendered
    assert "\n" not in rendered
    assert "\r" not in rendered


def test_artifact_format_error_defaults_to_invalid_shape():
    error = ArtifactFormatError("bad artifact")

    assert error.code == "io.invalid_artifact_shape"


def test_json_gz_decode_errors_preserve_cause(tmp_path):
    path = tmp_path / "bad.json.gz"
    path.write_bytes(b"not gzip")

    with pytest.raises(ArtifactDecodeError) as exc_info:
        load_json_gz(path)

    assert exc_info.value.code == "io.decode_failed"
    assert exc_info.value.__cause__ is not None


def test_benchmark_schema_errors_preserve_cause(tmp_path):
    path = tmp_path / "benchmark.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"name": "broken", "targets": {"x": {"id": "x"}}}, f)

    with pytest.raises(ArtifactFormatError) as exc_info:
        load_benchmark(path)

    assert exc_info.value.code == "io.invalid_artifact_shape"
    assert exc_info.value.context["artifact"] == "benchmark"
    assert exc_info.value.__cause__ is not None


def test_no_direct_retrocast_exception_raises_outside_exceptions_module():
    offenders = []
    for path in _python_files():
        if path.name == "exceptions.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                if isinstance(func, ast.Name) and func.id == "RetroCastException":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []


def test_broad_exception_catches_stay_at_boundaries():
    allowed = {
        Path("src/retrocast/chem.py"),
        Path("src/retrocast/cli/adhoc.py"),
        Path("src/retrocast/cli/compare.py"),
        Path("src/retrocast/cli/handlers.py"),
        Path("src/retrocast/cli/main.py"),
        Path("src/retrocast/io/data.py"),
        Path("src/retrocast/utils/serializers.py"),
        Path("src/retrocast/workflow/verify.py"),
    }
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            broad = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
            )
            if broad and rel not in allowed:
                offenders.append(f"{rel}:{node.lineno}")

    assert offenders == []


def test_sys_exit_is_cli_only():
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sys"
                and (rel.parts[:2] != ("src", "retrocast") or rel.parts[2] != "cli")
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert offenders == []
