import argparse
from pathlib import Path

import pytest
import yaml
from pytest_mock import MockerFixture

from retrocast.cli import (
    _run_single_process,
    get_model_config,
    handle_info,
    handle_list,
    handle_process,
    load_config,
)
from retrocast.exceptions import RetroCastException


@pytest.fixture
def mock_config() -> dict:
    """Provides a standard mock config dictionary."""
    return {
        "models": {
            "model-a": {
                "adapter": "a_adapter",
                "raw_results_filename": "results-a.json.gz",
                "preprocess_script": "scripts/preprocess_a.py",
                "sampling": {"strategy": "top-k", "k": 10},
            },
            "model-b": {
                "adapter": "b_adapter",
                "raw_results_filename": "results-b.json.gz",
            },
        }
    }


@pytest.fixture
def mock_project_structure(tmp_path: Path, mock_config: dict) -> Path:
    """Creates a mock project directory structure with necessary files."""
    base_dir = tmp_path / "project"
    base_dir.mkdir()

    # Config file
    config_path = base_dir / "retrocast-config.yaml"
    with config_path.open("w") as f:
        yaml.dump(mock_config, f)

    # Data directories
    data_dir = base_dir / "data"
    (data_dir / "targets").mkdir(parents=True)
    (data_dir / "evaluations" / "model-a" / "dataset-1").mkdir(parents=True)
    (data_dir / "evaluations" / "model-b" / "dataset-2").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)

    # Dummy files
    (data_dir / "targets" / "dataset-1.csv").touch()
    (data_dir / "targets" / "dataset-2.csv").touch()
    (data_dir / "evaluations" / "model-a" / "dataset-1" / "results-a.json.gz").touch()
    (data_dir / "evaluations" / "model-b" / "dataset-2" / "results-b.json.gz").touch()

    return base_dir


class TestCliUnit:
    def test_load_config_success(self, mock_project_structure: Path):
        """Tests that a valid config file is loaded correctly."""
        config = load_config(mock_project_structure / "retrocast-config.yaml")
        assert "models" in config
        assert "model-a" in config["models"]

    def test_load_config_not_found(self, tmp_path: Path):
        """Tests that a missing config file causes a sys.exit."""
        with pytest.raises(SystemExit) as e:
            load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(e.value, SystemExit)
        assert e.value.code == 1

    def test_load_config_yaml_error(self, tmp_path: Path):
        """Tests that a malformed yaml file causes a sys.exit."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [value")  # malformed yaml
        with pytest.raises(SystemExit) as e:
            load_config(bad_yaml)
        assert isinstance(e.value, SystemExit)
        assert e.value.code == 1

    def test_get_model_config_success(self, mock_config: dict):
        """Tests successful retrieval of a model's config."""
        model_conf = get_model_config(mock_config, "model-a")
        assert model_conf["adapter"] == "a_adapter"

    def test_get_model_config_not_found(self, mock_config: dict):
        """Tests that a missing model name causes a sys.exit."""
        with pytest.raises(SystemExit) as e:
            get_model_config(mock_config, "model-c")
        assert isinstance(e.value, SystemExit)
        assert e.value.code == 1

    def test_handle_list(self, mock_config: dict, capsys):
        """Tests that the 'list' command prints the available models."""
        handle_list(mock_config)
        captured = capsys.readouterr()
        assert "- model-a" in captured.out
        assert "- model-b" in captured.out

    def test_handle_info_with_all_fields(self, mock_config: dict, caplog, capsys):
        """Tests 'info' for a model with a preprocess script and sampling."""
        handle_info(mock_config, "model-a")
        captured_out = capsys.readouterr().out
        captured_log = caplog.text
        assert "pre-processing required" in captured_log
        assert "uv run scripts/preprocess_a.py" in captured_out
        assert "default sampling: strategy='top-k', k=10" in captured_log

    def test_handle_info_with_minimal_fields(self, mock_config: dict, caplog):
        """Tests 'info' for a model with no preprocess script or sampling."""
        handle_info(mock_config, "model-b")
        assert "no pre-processing script defined" in caplog.text
        assert "default sampling: none" in caplog.text
        assert "expected raw results filename: results-b.json.gz" in caplog.text


class TestCliProcessExecution:
    @pytest.fixture
    def mock_dependencies(self, mocker: MockerFixture) -> dict:
        """Mocks all functions called by _run_single_process."""
        return {
            "load_targets": mocker.patch("retrocast.cli.load_and_prepare_targets", return_value={"target": "info"}),
            "get_adapter": mocker.patch("retrocast.cli.get_adapter"),
            "process_run": mocker.patch("retrocast.cli.process_model_run"),
        }

    def test_run_single_process_happy_path(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict
    ):
        """Tests a successful run with config-defined sampling."""
        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_dependencies["process_run"].assert_called_once()
        call_args = mock_dependencies["process_run"].call_args[1]
        assert call_args["model_name"] == "model-a"
        assert call_args["dataset_name"] == "dataset-1"
        assert call_args["sampling_strategy"] == "top-k"
        assert call_args["sample_k"] == 10

    def test_run_single_process_with_overrides(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict
    ):
        """Tests that CLI overrides for sampling are respected."""
        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", "random-k", 20)
        call_args = mock_dependencies["process_run"].call_args[1]
        assert call_args["sampling_strategy"] == "random-k"
        assert call_args["sample_k"] == 20

    def test_run_single_process_raw_file_not_found(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict, caplog
    ):
        """Tests that a missing raw results file is handled gracefully."""
        # delete the file
        raw_file = mock_project_structure / "data" / "evaluations" / "model-a" / "dataset-1" / "results-a.json.gz"
        raw_file.unlink()

        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_dependencies["process_run"].assert_not_called()
        assert f"raw results file not found: {raw_file}" in caplog.text
        assert "did you run the pre-processing script" in caplog.text

    def test_run_single_process_targets_file_not_found(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict, caplog
    ):
        """Tests that a missing targets file is handled gracefully."""
        targets_file = mock_project_structure / "data" / "targets" / "dataset-1.csv"
        targets_file.unlink()

        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_dependencies["process_run"].assert_not_called()
        assert f"targets file not found: {targets_file}" in caplog.text

    def test_run_single_process_handles_ursa_exception(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict, caplog
    ):
        """Tests that RetroCastExceptions are caught and logged."""
        mock_dependencies["load_targets"].side_effect = RetroCastException("Bad SMILES")
        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_dependencies["process_run"].assert_not_called()
        assert "a critical error occurred" in caplog.text
        assert "Bad SMILES" in caplog.text

    def test_run_single_process_handles_unexpected_exception(
        self, mock_project_structure: Path, mock_config: dict, mock_dependencies: dict, caplog
    ):
        """Tests that unexpected exceptions are caught and logged as critical."""
        mock_dependencies["load_targets"].side_effect = ValueError("Something unexpected")
        _run_single_process(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_dependencies["process_run"].assert_not_called()
        assert "an unexpected error occurred" in caplog.text
        assert "Something unexpected" in caplog.text


class TestHandleProcessDispatch:
    @pytest.fixture
    def mock_run_single(self, mocker: MockerFixture):
        """Mocks the core _run_single_process function."""
        return mocker.patch("retrocast.cli._run_single_process")

    def test_handle_process_single_run(self, mock_project_structure, mock_config, mock_run_single):
        """Tests dispatching a single model/dataset run."""
        args = argparse.Namespace(
            model="model-a",
            dataset="dataset-1",
            all_models=False,
            all_datasets=False,
            sampling_strategy=None,
            k=None,
        )
        handle_process(mock_project_structure, mock_config, args)
        mock_run_single.assert_called_once_with(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)

    def test_handle_process_all_models(self, mock_project_structure, mock_config, mock_run_single):
        """Tests --all-models flag."""
        args = argparse.Namespace(
            model=None,
            dataset="dataset-1",
            all_models=True,
            all_datasets=False,
            sampling_strategy=None,
            k=None,
        )
        handle_process(mock_project_structure, mock_config, args)
        assert mock_run_single.call_count == 2
        mock_run_single.assert_any_call(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_run_single.assert_any_call(mock_project_structure, mock_config, "model-b", "dataset-1", None, None)

    def test_handle_process_all_datasets(self, mock_project_structure, mock_config, mock_run_single):
        """Tests --all-datasets flag."""
        args = argparse.Namespace(
            model="model-a",
            dataset=None,
            all_models=False,
            all_datasets=True,
            sampling_strategy=None,
            k=None,
        )
        handle_process(mock_project_structure, mock_config, args)
        assert mock_run_single.call_count == 2
        mock_run_single.assert_any_call(mock_project_structure, mock_config, "model-a", "dataset-1", None, None)
        mock_run_single.assert_any_call(mock_project_structure, mock_config, "model-a", "dataset-2", None, None)

    def test_handle_process_all_models_all_datasets(self, mock_project_structure, mock_config, mock_run_single):
        """Tests matrix run with --all-models and --all-datasets."""
        args = argparse.Namespace(
            model=None,
            dataset=None,
            all_models=True,
            all_datasets=True,
            sampling_strategy=None,
            k=None,
        )
        handle_process(mock_project_structure, mock_config, args)
        assert mock_run_single.call_count == 4  # 2 models * 2 datasets

    def test_handle_process_no_args_exits(self, mock_project_structure, mock_config, mock_run_single):
        """Tests that not specifying a model/dataset causes sys.exit."""
        args = argparse.Namespace(
            model=None, dataset=None, all_models=False, all_datasets=False, sampling_strategy=None, k=None
        )
        # We need to temporarily unpatch sys.exit for this test to work with pytest.raises
        with pytest.raises(SystemExit) as e:
            handle_process(mock_project_structure, mock_config, args)
        assert isinstance(e.value, SystemExit)
        assert e.value.code == 1
        mock_run_single.assert_not_called()
