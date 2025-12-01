"""
In-process integration tests for CLI handlers.
This executes the actual handler logic within the test process, ensuring coverage tracking.
"""

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from retrocast.chem import get_inchi_key
from retrocast.cli import handlers
from retrocast.io.blob import save_json_gz
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from tests.helpers import _synthetic_inchikey

# --- Fixtures ---


@pytest.fixture
def synthetic_config(tmp_path) -> dict[str, Any]:
    """
    Creates a full directory structure in tmp_path and returns a valid config dict.
    """
    base = tmp_path / "data"

    # Create structure
    (base / "1-benchmarks" / "definitions").mkdir(parents=True)
    (base / "1-benchmarks" / "stocks").mkdir(parents=True)
    (base / "2-raw" / "test-model" / "test-bench").mkdir(parents=True)
    (base / "3-processed").mkdir(parents=True)
    (base / "4-scored").mkdir(parents=True)
    (base / "5-results").mkdir(parents=True)

    return {
        "data_dir": str(base),
        "models": {"test-model": {"adapter": "paroutes", "raw_results_filename": "results.json.gz"}},
    }


@pytest.fixture
def synthetic_data(synthetic_config):
    """
    Populates the directories with valid synthetic files.
    """
    base = Path(synthetic_config["data_dir"])

    # 1. Create Benchmark
    target = BenchmarkTarget(
        id="t1",
        smiles="CC",  # Ethane - matches our route target
        inchi_key=_synthetic_inchikey("CC"),
        is_convergent=True,  # Two reactants merge in one reaction
        route_length=1,  # One reaction step
        ground_truth=None,
    )
    bench = BenchmarkSet(name="test-bench", stock_name="test-stock", targets={"t1": target})
    save_json_gz(bench, base / "1-benchmarks" / "definitions" / "test-bench.json.gz")

    # 2. Create Stock
    # Stock contains methane (C), which is the reactant in our route
    stock_path = base / "1-benchmarks" / "stocks" / "test-stock.csv.gz"
    with gzip.open(stock_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES", "InChIKey"])
        writer.writerow(["C", get_inchi_key("C")])

    # 3. Create Raw Results
    # ParoutesAdapter expects a molecule node with reaction children
    # Creating a simple route: CC <- C + C (ethane from two methane molecules)

    # Leaf molecules (reactants)
    reactant1 = {"type": "mol", "smiles": "C", "in_stock": True, "children": []}

    reactant2 = {"type": "mol", "smiles": "C", "in_stock": True, "children": []}

    # Reaction node combining the reactants
    reaction_node = {
        "type": "reaction",
        "smiles": "CC",  # Product SMILES
        "metadata": {"ID": "US2020123456A1;example-rxn", "rsmi": "C.C>>CC"},
        "children": [reactant1, reactant2],
    }

    # Root molecule (target) with the reaction as a child
    raw_route_tree = {"type": "mol", "smiles": "CC", "in_stock": False, "children": [reaction_node]}

    # Map target_id -> Single Route Dict (not list)
    raw_data_map = {"t1": raw_route_tree}

    save_json_gz(raw_data_map, base / "2-raw" / "test-model" / "test-bench" / "results.json.gz")

    return bench


# --- Tests ---


@pytest.mark.integration
class TestCLIHandlerIntegration:
    """Integration tests for the full CLI handler workflow.

    Tests the complete pipeline: ingest -> score -> analyze
    using synthetic data with tmp_path.
    """

    def test_ingest_flow(self, synthetic_config, synthetic_data):
        """Test handle_ingest -> creates processed routes."""

        # Mock CLI args
        args = SimpleNamespace(
            model="test-model",
            dataset="test-bench",
            all_models=False,
            all_datasets=False,
            sampling_strategy=None,
            k=None,
            anonymize=False,
        )

        # RUN
        handlers.handle_ingest(args, synthetic_config)

        # VERIFY
        base = Path(synthetic_config["data_dir"])
        expected_file = base / "3-processed" / "test-bench" / "test-model" / "routes.json.gz"
        assert expected_file.exists()

        # Check manifest exists
        assert (expected_file.parent / "manifest.json").exists()

    def test_score_flow(self, synthetic_config, synthetic_data):
        """Test handle_score -> creates evaluation file."""

        # Pre-requisite: Run ingest first
        self.test_ingest_flow(synthetic_config, synthetic_data)

        args = SimpleNamespace(
            model="test-model", dataset="test-bench", all_models=False, all_datasets=False, stock=None
        )

        # RUN
        handlers.handle_score(args, synthetic_config)

        # VERIFY
        base = Path(synthetic_config["data_dir"])
        expected_file = base / "4-scored" / "test-bench" / "test-model" / "test-stock" / "evaluation.json.gz"
        assert expected_file.exists()

        # Load check
        with gzip.open(expected_file, "rt") as f:
            data = json.load(f)
            # t1 should be solvable because stock has "C" and route uses C as reactants
            assert data["results"]["t1"]["is_solvable"] is True

    def test_analyze_flow(self, synthetic_config, synthetic_data):
        """Test handle_analyze -> creates report and plots."""

        # Pre-requisite: Run scoring first
        self.test_score_flow(synthetic_config, synthetic_data)

        args = SimpleNamespace(
            model="test-model",
            dataset="test-bench",
            all_models=False,
            all_datasets=False,
            stock=None,
            make_plots=False,
            top_k=[1, 5, 10],
        )

        # RUN
        handlers.handle_analyze(args, synthetic_config)

        # VERIFY
        base = Path(synthetic_config["data_dir"])
        results_dir = base / "5-results" / "test-bench" / "test-model" / "test-stock"

        assert (results_dir / "statistics.json.gz").exists()
        assert (results_dir / "report.md").exists()

    def test_missing_file_handling(self, synthetic_config, synthetic_data, caplog):
        """Ensure handlers fail gracefully when raw files are missing."""

        # FIX: Added synthetic_data fixture above so the benchmark definition exists.
        # Now we manually delete the raw file to test that specific failure mode.

        base = Path(synthetic_config["data_dir"])
        raw_file = base / "2-raw" / "test-model" / "test-bench" / "results.json.gz"
        if raw_file.exists():
            raw_file.unlink()

        args = SimpleNamespace(
            model="test-model",
            dataset="test-bench",
            all_models=False,
            all_datasets=False,
            sampling_strategy=None,
            k=None,
            anonymize=False,
        )

        # Should simply return/log warning, not raise FileNotFoundError
        handlers.handle_ingest(args, synthetic_config)

        assert "File not found" in caplog.text or "Skipping" in caplog.text
