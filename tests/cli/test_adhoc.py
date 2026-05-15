"""
Integration tests for CLI adhoc handlers.

Tests follow the testing framework philosophy:
- No mocking (use tmp_path for file I/O)
- Simple synthetic data (C, CC, CCC for testing)
- Integration tests for end-to-end handler behavior
"""

import json
from argparse import Namespace
from collections.abc import Generator
from typing import Any

import pytest

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.cli.adhoc import handle_adapt, handle_collect, handle_create_benchmark
from retrocast.exceptions import ChemRuntimeError
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import load_route_corpus, save_route_corpus
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr
from tests.helpers import _make_simple_route, _synthetic_inchikey


class ChemFailingAdapter(BaseAdapter):
    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
        expected_target: TargetIdentity | None = None,
    ) -> Generator[RawRouteEntry, None, None]:
        if isinstance(raw_data, dict):
            for key, payload in raw_data.items():
                yield RawRouteEntry(payload=payload, source_key=str(key))
            return
        yield RawRouteEntry(payload=raw_data, source_key=source_key)

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        raise ChemRuntimeError(
            "synthetic chemistry failure",
            context={"target_id": expected_target.id if expected_target else None},
        )


@pytest.mark.integration
class TestHandleCreateBenchmark:
    """Integration tests for handle_create_benchmark handler."""

    def test_create_benchmark_from_csv_basic(self, tmp_path):
        """Test basic CSV with id and smiles columns."""
        # Create input CSV
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("id,smiles\ntarget-001,C\ntarget-002,CC\ntarget-003,CCC\n")

        output_base = tmp_path / "test-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="test-benchmark",
            stock_name="test-stock",
        )

        handle_create_benchmark(args)

        # Verify output files exist
        output_path = tmp_path / "test-benchmark.json.gz"
        manifest_path = tmp_path / "test-benchmark.manifest.json"
        assert output_path.exists()
        assert manifest_path.exists()

        # Load and verify benchmark
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        assert benchmark.name == "test-benchmark"
        assert benchmark.stock_name == "test-stock"
        assert len(benchmark.targets) == 3
        assert "target-001" in benchmark.targets
        assert benchmark.targets["target-001"].smiles == "C"
        assert benchmark.targets["target-002"].smiles == "CC"
        assert benchmark.targets["target-003"].smiles == "CCC"

        # Verify no acceptable routes (no ground truth)
        assert benchmark.targets["target-001"].acceptable_routes == []
        assert benchmark.targets["target-001"].primary_route is None

    def test_create_benchmark_from_csv_flexible_columns(self, tmp_path):
        """Test CSV with flexible column names (SMILES, Target ID)."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("Target ID,SMILES\nmol-1,C\nmol-2,CC\n")

        output_base = tmp_path / "flexible-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="flexible-test",
            stock_name=None,
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "flexible-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        assert len(benchmark.targets) == 2
        assert "mol-1" in benchmark.targets
        assert "mol-2" in benchmark.targets
        assert benchmark.stock_name is None

    def test_create_benchmark_from_csv_with_metadata(self, tmp_path):
        """Test CSV with extra metadata columns."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("id,smiles,source,complexity\nt1,C,literature,easy\nt2,CC,patent,medium\n")

        output_base = tmp_path / "meta-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="meta-test",
            stock_name="stock-v1",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "meta-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        # Verify metadata is captured
        assert benchmark.targets["t1"].metadata["source"] == "literature"
        assert benchmark.targets["t1"].metadata["complexity"] == "easy"
        assert benchmark.targets["t2"].metadata["source"] == "patent"
        assert benchmark.targets["t2"].metadata["complexity"] == "medium"

    def test_create_benchmark_from_csv_alternative_names(self, tmp_path):
        """Test CSV with various acceptable column name variants."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("structure_id,smi\nx1,C\nx2,O\n")

        output_base = tmp_path / "alt-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="alt-test",
            stock_name="stock",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "alt-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        assert len(benchmark.targets) == 2
        assert "x1" in benchmark.targets
        assert benchmark.targets["x1"].smiles == "C"
        assert benchmark.targets["x2"].smiles == "O"

    def test_create_benchmark_from_txt_basic(self, tmp_path):
        """Test TXT file with one SMILES per line."""
        txt_path = tmp_path / "targets.txt"
        txt_path.write_text("C\nCC\nCCC\nCCCC\n")

        output_base = tmp_path / "txt-benchmark"
        args = Namespace(
            input=str(txt_path),
            output=str(output_base),
            name="txt-test",
            stock_name="stock",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "txt-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        assert len(benchmark.targets) == 4
        # Auto-generated IDs
        assert "target-1" in benchmark.targets
        assert "target-2" in benchmark.targets
        assert "target-3" in benchmark.targets
        assert "target-4" in benchmark.targets
        assert benchmark.targets["target-1"].smiles == "C"
        assert benchmark.targets["target-4"].smiles == "CCCC"

    def test_create_benchmark_from_txt_with_blank_lines(self, tmp_path):
        """Test TXT file handles blank lines correctly."""
        txt_path = tmp_path / "targets.txt"
        txt_path.write_text("C\n\nCC\n  \nCCC\n")

        output_base = tmp_path / "txt-blank-benchmark"
        args = Namespace(
            input=str(txt_path),
            output=str(output_base),
            name="txt-blank-test",
            stock_name="stock",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "txt-blank-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        # Should skip blank lines
        assert len(benchmark.targets) == 3

    def test_create_benchmark_from_txt_id_padding(self, tmp_path):
        """Test TXT file auto-generates IDs with correct padding."""
        txt_path = tmp_path / "targets.txt"
        # Create 12 distinct targets to test zero-padding
        # Use different alkanes: C, CC, CCC, etc.
        smiles_list = ["C" * (i + 1) for i in range(12)]
        txt_path.write_text("\n".join(smiles_list))

        output_base = tmp_path / "txt-padding-benchmark"
        args = Namespace(
            input=str(txt_path),
            output=str(output_base),
            name="padding-test",
            stock_name="stock",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "txt-padding-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        # Check padding (12 targets = 2 digits)
        assert "target-01" in benchmark.targets
        assert "target-12" in benchmark.targets
        assert "target-1" not in benchmark.targets  # Should be padded

    def test_create_benchmark_missing_input_file(self, tmp_path):
        """Test handler exits when input file doesn't exist."""
        output_base = tmp_path / "output"
        args = Namespace(
            input="/nonexistent/file.csv",
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_unsupported_extension(self, tmp_path):
        """Test handler exits with unsupported file extension."""
        bad_path = tmp_path / "targets.json"
        bad_path.write_text("{}")

        output_base = tmp_path / "output"
        args = Namespace(
            input=str(bad_path),
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_csv_missing_smiles_column(self, tmp_path):
        """Test handler exits when CSV lacks SMILES column."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("id,molecule\ntarget-1,C\n")

        output_base = tmp_path / "output"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_csv_missing_id_column(self, tmp_path):
        """Test handler exits when CSV lacks ID column."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("name,smiles\nmol,C\n")

        output_base = tmp_path / "output"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_csv_empty_file(self, tmp_path):
        """Test handler exits with empty CSV."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("")

        output_base = tmp_path / "output"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_invalid_smiles(self, tmp_path):
        """Test handler exits with invalid SMILES."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("id,smiles\nt1,INVALID_SMILES_XXX\n")

        output_base = tmp_path / "output"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="test",
            stock_name="stock",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_canonicalization(self, tmp_path):
        """Test SMILES are canonicalized and duplicates are detected."""
        csv_path = tmp_path / "targets.csv"
        # Non-canonical SMILES for different molecules
        # CCO = ethanol, c1ccccc1 = benzene (non-canonical)
        csv_path.write_text("id,smiles\nt1,CCO\nt2,c1ccccc1\n")

        output_base = tmp_path / "canon-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="canon-test",
            stock_name="stock",
        )

        handle_create_benchmark(args)

        output_path = tmp_path / "canon-benchmark.json.gz"
        data = load_json_gz(output_path)
        benchmark = BenchmarkSet.model_validate(data)

        # Should be canonicalized
        assert benchmark.targets["t1"].smiles == "CCO"
        assert benchmark.targets["t2"].smiles == "c1ccccc1"  # Canonical form

    def test_create_benchmark_rejects_duplicate_smiles(self, tmp_path):
        """Test that duplicate SMILES (after canonicalization) are rejected."""
        csv_path = tmp_path / "targets.csv"
        # Non-canonical SMILES: ethanol as CCO vs OCC (both canonicalize to CCO)
        csv_path.write_text("id,smiles\nt1,CCO\nt2,OCC\n")

        output_base = tmp_path / "canon-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="canon-test",
            stock_name="stock",
        )

        # Should fail due to duplicate SMILES
        with pytest.raises(SystemExit) as exc_info:
            handle_create_benchmark(args)
        assert exc_info.value.code == 1

    def test_create_benchmark_manifest_content(self, tmp_path):
        """Test manifest contains expected metadata."""
        csv_path = tmp_path / "targets.csv"
        csv_path.write_text("id,smiles\nt1,C\nt2,CC\n")

        output_base = tmp_path / "manifest-benchmark"
        args = Namespace(
            input=str(csv_path),
            output=str(output_base),
            name="manifest-test",
            stock_name="test-stock",
        )

        handle_create_benchmark(args)

        manifest_path = tmp_path / "manifest-benchmark.manifest.json"
        import json

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["action"] == "[cli]create-benchmark"
        assert manifest["parameters"]["name"] == "manifest-test"
        assert manifest["parameters"]["stock_name"] == "test-stock"
        assert manifest["statistics"]["n_targets"] == 2
        assert len(manifest["source_files"]) == 1
        assert len(manifest["output_files"]) == 1


@pytest.mark.integration
class TestHandleAdapt:
    def test_adapt_skips_target_local_chem_failures(self, tmp_path, monkeypatch):
        input_path = tmp_path / "raw.json.gz"
        output_path = tmp_path / "route-corpus.jsonl.gz"
        save_json_gz({"CCO": {"bad": "payload"}}, input_path)
        monkeypatch.setattr("retrocast.cli.adhoc.get_adapter", lambda name: ChemFailingAdapter())

        args = Namespace(
            input=str(input_path),
            output=str(output_path),
            adapter="synthetic",
            benchmark=None,
        )

        handle_adapt(args)

        assert load_route_corpus(output_path) == []

    def test_adapt_loads_flat_jsonl_corpora(self, tmp_path):
        input_path = tmp_path / "completions.jsonl"
        output_path = tmp_path / "route-corpus.jsonl.gz"
        record = {
            "meta": {"product_smiles": "CC(=O)C"},
            "completion": (
                "<synthesis_step><product><smiles>CC(=O)C</smiles></product>"
                "<reactant><smiles>CC(=O)O</smiles></reactant>"
                "<reactant><smiles>C</smiles></reactant></synthesis_step>"
            ),
        }
        input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        args = Namespace(
            input=str(input_path),
            output=str(output_path),
            adapter="ursa-llm",
            benchmark=None,
        )

        handle_adapt(args)

        route_corpus = load_route_corpus(output_path)
        assert len(route_corpus) == 1
        assert route_corpus[0].target.smiles == "CC(C)=O"

    def test_collect_builds_benchmark_keyed_routes(self, tmp_path):
        input_path = tmp_path / "route-corpus.jsonl.gz"
        benchmark_path = tmp_path / "benchmark.json.gz"
        output_path = tmp_path / "routes.json.gz"

        save_route_corpus([_make_simple_route("CC", "C", rank=7)], input_path)

        benchmark = BenchmarkSet(
            name="collect-test",
            targets={
                "target_1": BenchmarkTarget(
                    id="target_1",
                    smiles=SmilesStr("CC"),
                    inchi_key=_synthetic_inchikey("CC"),
                    acceptable_routes=[],
                )
            },
        )
        save_json_gz(benchmark.model_dump(mode="json"), benchmark_path)

        args = Namespace(
            input=str(input_path),
            benchmark=str(benchmark_path),
            output=str(output_path),
        )

        handle_collect(args)

        saved_routes = load_json_gz(output_path)
        assert list(saved_routes.keys()) == ["target_1"]
        assert len(saved_routes["target_1"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
