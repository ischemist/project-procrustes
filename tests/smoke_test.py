"""
Smoke tests for retrocast package.

These tests are run during PyPI publish to verify the wheel/sdist
contains all crucial files and core imports work. They must be
self-contained with NO external fixtures or test data dependencies.

Run with: pytest tests/smoke_test.py -v
"""

import pytest


class TestImports:
    """Test that core modules can be imported."""

    def test_import_retrocast(self):
        import retrocast

        assert retrocast is not None

    def test_import_version(self):
        from retrocast import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Version should be a valid semver-ish string
        assert len(__version__) > 0

    def test_import_api(self):
        from retrocast import api

        # Check key functions are exported
        assert hasattr(api, "load_benchmark")
        assert hasattr(api, "load_routes")
        assert hasattr(api, "load_stock_file")
        assert hasattr(api, "score_predictions")
        assert hasattr(api, "compute_metric_with_ci")

    def test_import_models(self):
        from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
        from retrocast.models.chem import Molecule, ReactionStep, Route
        from retrocast.models.evaluation import EvaluationResults, TargetEvaluation

        # Verify classes exist
        assert BenchmarkSet is not None
        assert BenchmarkTarget is not None
        assert Route is not None
        assert Molecule is not None
        assert ReactionStep is not None
        assert EvaluationResults is not None
        assert TargetEvaluation is not None

    def test_import_io(self):
        from retrocast.io import (
            load_benchmark,
            load_routes,
            load_stock_file,
            save_json_gz,
        )

        assert callable(load_benchmark)
        assert callable(load_routes)
        assert callable(load_stock_file)
        assert callable(save_json_gz)

    def test_import_adapters(self):
        from retrocast.adapters import get_adapter

        assert callable(get_adapter)

    def test_import_metrics(self):
        from retrocast.metrics.bootstrap import compute_metric_with_ci

        assert callable(compute_metric_with_ci)


class TestBasicFunctionality:
    """Test that basic objects can be created without errors."""

    def test_create_molecule(self):
        from retrocast.chem import get_inchi_key
        from retrocast.models.chem import Molecule

        mol = Molecule(smiles="C", inchikey=get_inchi_key("C"))
        assert mol.smiles == "C"
        assert mol.is_leaf is True

    def test_create_reaction_step(self):
        from retrocast.chem import get_inchi_key
        from retrocast.models.chem import Molecule, ReactionStep

        reactant = Molecule(smiles="C", inchikey=get_inchi_key("C"))
        step = ReactionStep(reactants=[reactant])
        assert len(step.reactants) == 1

    def test_create_route(self):
        from retrocast.chem import get_inchi_key
        from retrocast.models.chem import Molecule, Route

        mol = Molecule(smiles="CC", inchikey=get_inchi_key("CC"))
        route = Route(target=mol, rank=1)

        assert route.target.smiles == "CC"
        assert route.rank == 1

    def test_create_evaluation_results(self):
        from retrocast.models.evaluation import EvaluationResults

        results = EvaluationResults(
            model_name="test-model",
            benchmark_name="test-benchmark",
            stock_name="test-stock",
            has_acceptable_routes=False,
        )
        assert results.model_name == "test-model"
        assert len(results.results) == 0


class TestCLIEntryPoint:
    """Test that CLI entry point is accessible."""

    def test_cli_main_exists(self):
        from retrocast.cli.main import main

        assert callable(main)

    def test_cli_help(self):
        """Test that --help works without errors."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "retrocast.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "retrocast" in result.stdout.lower()


class TestChemistry:
    """Test basic chemistry utilities."""

    def test_canonicalize_smiles(self):
        from retrocast.chem import canonicalize_smiles

        # Simple canonicalization
        result = canonicalize_smiles("C")
        assert result == "C"

        # Should handle ethane
        result = canonicalize_smiles("CC")
        assert result == "CC"

    def test_smiles_validation_via_canonicalize(self):
        from retrocast.chem import canonicalize_smiles
        from retrocast.exceptions import InvalidSmilesError

        # Valid SMILES should work
        assert canonicalize_smiles("C") == "C"
        assert canonicalize_smiles("CC") == "CC"

        # Invalid SMILES should raise
        with pytest.raises(InvalidSmilesError):
            canonicalize_smiles("invalid_smiles_xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
