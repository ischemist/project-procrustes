from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from retrocast.chem import canonicalize_smiles
from retrocast.io.blob import load_json_gz

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/llm-raw-answers/1-convert-to-json.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("llm_raw_answers_preprocess", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
class TestLlmRawAnswersPreprocessScript:
    def test_main_writes_canonical_target_keys_and_truthful_summary(self, tmp_path, monkeypatch):
        raw_target_smiles = "C1=CC=CC=C1"
        canonical_target_smiles = canonicalize_smiles(raw_target_smiles)
        input_path = tmp_path / "completions.jsonl"
        output_dir = tmp_path / "converted"
        records = [
            {"meta": {"product_smiles": raw_target_smiles}, "completion": "route-1"},
            {"meta": {"product_smiles": canonical_target_smiles}, "completion": "route-2"},
            {"meta": "not-a-dict", "completion": "route-3"},
            {"meta": {"product_smiles": "not-a-smiles"}, "completion": "route-4"},
            "not-a-dict",
        ]
        input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT_PATH),
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module()
        script_module.main()

        results = load_json_gz(output_dir / "results.json.gz")
        summary = json.loads((output_dir / "summary.json").read_text())

        assert raw_target_smiles not in results
        assert results == {
            canonical_target_smiles: [
                {"completion": "route-1"},
                {"completion": "route-2"},
            ]
        }
        assert summary == {
            "solved_count": 1,
            "total_records": 5,
            "accepted_records": 2,
            "skipped_records": 3,
        }

    def test_main_exits_nonzero_when_input_is_missing(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "converted"
        missing_input = tmp_path / "missing.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT_PATH),
                "--input",
                str(missing_input),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module()
        with pytest.raises(SystemExit) as exc_info:
            script_module.main()

        assert exc_info.value.code == 1

    def test_main_exits_nonzero_when_input_is_invalid_json(self, tmp_path, monkeypatch):
        input_path = tmp_path / "bad.jsonl"
        output_dir = tmp_path / "converted"
        input_path.write_text("{this is not json}\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT_PATH),
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module()
        with pytest.raises(SystemExit) as exc_info:
            script_module.main()

        assert exc_info.value.code == 1
