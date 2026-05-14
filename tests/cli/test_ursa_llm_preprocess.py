from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from retrocast.adapters.ursa_llm_adapter import prepare_ursa_llm_results
from retrocast.chem import canonicalize_smiles
from retrocast.io.blob import load_json_gz

CANONICAL_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/ursa-llm/1-prepare-raw-results.py"


def _load_script_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
class TestUrsaLlmPreprocessScript:
    def test_main_writes_canonical_target_keys_and_truthful_summary(
        self, tmp_path, monkeypatch
    ):
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
                str(CANONICAL_SCRIPT_PATH),
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module(CANONICAL_SCRIPT_PATH, "ursa_llm_prepare_raw_results")
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
                str(CANONICAL_SCRIPT_PATH),
                "--input",
                str(missing_input),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module(CANONICAL_SCRIPT_PATH, "ursa_llm_prepare_raw_results_missing")
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
                str(CANONICAL_SCRIPT_PATH),
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
            ],
        )

        script_module = _load_script_module(CANONICAL_SCRIPT_PATH, "ursa_llm_prepare_raw_results_invalid")
        with pytest.raises(SystemExit) as exc_info:
            script_module.main()

        assert exc_info.value.code == 1


@pytest.mark.integration
@pytest.mark.parametrize("suffix", [".json.gz", ".jsonl.gz"])
def test_prepare_ursa_llm_results_supports_gzipped_inputs(tmp_path, suffix):
    raw_target_smiles = "C1=CC=CC=C1"
    records = [
        {"meta": {"product_smiles": raw_target_smiles}, "completion": "route-1"},
        {"meta": {"product_smiles": "not-a-smiles"}, "completion": "route-2"},
    ]
    input_path = tmp_path / f"completions{suffix}"

    if suffix == ".json.gz":
        with gzip.open(input_path, "wt", encoding="utf-8") as handle:
            json.dump(records, handle)
    else:
        with gzip.open(input_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    results, summary = prepare_ursa_llm_results(input_path)

    canonical_target_smiles = canonicalize_smiles(raw_target_smiles)
    assert results == {canonical_target_smiles: [{"completion": "route-1"}]}
    assert summary == {
        "solved_count": 1,
        "total_records": 2,
        "accepted_records": 1,
        "skipped_records": 1,
    }
