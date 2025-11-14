import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.core import process_model_run
from retrocast.domain.schemas import BenchmarkTree, MoleculeNode, TargetInfo
from retrocast.exceptions import UrsaIOException
from retrocast.io import save_json_gz
from retrocast.typing import SmilesStr
from retrocast.utils.hashing import generate_model_hash


@pytest.fixture
def aspirin_target_info() -> TargetInfo:
    """Provides a standard TargetInfo object for aspirin."""
    return TargetInfo(
        id="aspirin",
        smiles=SmilesStr("CC(=O)OC1=CC=CC=C1C(=O)O"),
    )


@pytest.fixture
def minimal_fake_tree(aspirin_target_info: TargetInfo) -> BenchmarkTree:
    """Provides a minimal, valid BenchmarkTree for mocking adapter outputs."""
    root_node = MoleculeNode(
        id="ursa-mol-root",
        molecule_hash="hash_aspirin",
        smiles=aspirin_target_info.smiles,
        is_starting_material=True,
        reactions=[],
    )
    return BenchmarkTree(
        target=aspirin_target_info,
        retrosynthetic_tree=root_node,
    )


@pytest.fixture
def multiple_unique_trees(minimal_fake_tree: BenchmarkTree) -> list[BenchmarkTree]:
    """Provides a list of three semantically unique BenchmarkTree objects."""
    tree1 = minimal_fake_tree.model_copy(deep=True)
    tree2 = minimal_fake_tree.model_copy(deep=True)
    tree3 = minimal_fake_tree.model_copy(deep=True)

    tree1.retrosynthetic_tree.molecule_hash = "hash_1"
    tree2.retrosynthetic_tree.molecule_hash = "hash_2"
    tree3.retrosynthetic_tree.molecule_hash = "hash_3"

    return [tree1, tree2, tree3]


def test_process_model_run_no_sampling(
    tmp_path: Path, mocker: MockerFixture, aspirin_target_info: TargetInfo, multiple_unique_trees: list[BenchmarkTree]
) -> None:
    """
    Tests the full orchestration without any sampling. Verifies correct output
    and that 'sampling_parameters' key is absent from the manifest.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_v1", "test_dataset"
    raw_file_content = {"aspirin": [{"smiles": "...", "children": []}]}
    raw_file_path = raw_dir / "target_aspirin.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.adapt.return_value = iter(multiple_unique_trees)

    # ACT
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
        sampling_strategy=None,
        sample_k=None,
    )

    # ASSERT
    expected_model_hash = generate_model_hash(model_name)
    output_dir = processed_dir / expected_model_hash
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    assert "sampling_parameters" not in manifest
    assert manifest["statistics"]["final_unique_routes_saved"] == len(multiple_unique_trees)


@pytest.mark.parametrize(
    "strategy, k, saved_routes",
    [
        ("top_k", 2, 2),
        ("random_k", 1, 1),
        ("by_length", 2, 2),
    ],
)
def test_process_model_run_with_sampling_strategies(
    strategy: str,
    k: int,
    saved_routes: int,
    tmp_path: Path,
    mocker: MockerFixture,
    aspirin_target_info: TargetInfo,
    multiple_unique_trees: list[BenchmarkTree],
) -> None:
    """
    Tests that the correct sampling function is called via the strategy map and
    that parameters are recorded in the manifest.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = f"model_{strategy}", f"data_{strategy}"
    raw_file_content = {"aspirin": [{"smiles": "...", "children": []}]}
    raw_file_path = raw_dir / "target_aspirin.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.adapt.return_value = iter(multiple_unique_trees)

    mock_sampling_func = mocker.MagicMock()
    mock_sampling_func.return_value = multiple_unique_trees[:saved_routes]
    mocker.patch.dict("retrocast.core.SAMPLING_STRATEGY_MAP", {strategy: mock_sampling_func})

    # ACT
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
        sampling_strategy=strategy,
        sample_k=k,
    )

    # ASSERT
    mock_sampling_func.assert_called_once_with(multiple_unique_trees, k)

    expected_model_hash = generate_model_hash(model_name)
    output_dir = processed_dir / expected_model_hash
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "Manifest was not created."

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    assert "sampling_parameters" in manifest
    assert manifest["sampling_parameters"]["strategy"] == strategy
    assert manifest["sampling_parameters"]["k"] == k
    assert manifest["statistics"]["final_unique_routes_saved"] == saved_routes


def test_process_model_run_handles_io_error(
    tmp_path: Path, mocker: MockerFixture, caplog, aspirin_target_info: TargetInfo
) -> None:
    """
    tests that a fatal error is logged and execution is aborted if the
    raw results file cannot be loaded.
    """
    # arrange
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_io_error", "test_dataset"
    raw_file_path = raw_dir / "results.json.gz"
    raw_file_path.touch()

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    # mock load_json_gz to simulate a file read/parse failure
    mocker.patch("retrocast.core.load_json_gz", side_effect=UrsaIOException("test error"))

    # act
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
    )

    # assert
    expected_model_hash = generate_model_hash(model_name)
    output_dir = processed_dir / expected_model_hash
    manifest_path = output_dir / "manifest.json"

    assert not manifest_path.exists()
    assert "fatal: could not read or parse input file" in caplog.text.lower()


def test_process_model_run_warns_on_missing_k(
    tmp_path: Path,
    mocker: MockerFixture,
    caplog,
    aspirin_target_info: TargetInfo,
    multiple_unique_trees: list[BenchmarkTree],
) -> None:
    """
    tests that a warning is logged if a sampling strategy is given but k is not.
    """
    # arrange
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_missing_k", "test_dataset"
    raw_file_path = raw_dir / "results.json.gz"
    save_json_gz({"aspirin": [{}]}, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.adapt.return_value = iter(multiple_unique_trees)

    # act
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
        sampling_strategy="top-k",
        sample_k=None,  # the key part of this test
    )

    # assert
    assert "specified but 'sample_k' is not set" in caplog.text
    # verify no sampling was actually applied
    manifest_path = processed_dir / generate_model_hash(model_name) / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    assert manifest["statistics"]["final_unique_routes_saved"] == len(multiple_unique_trees)


def test_process_model_run_warns_on_unknown_strategy(
    tmp_path: Path,
    mocker: MockerFixture,
    caplog,
    aspirin_target_info: TargetInfo,
    multiple_unique_trees: list[BenchmarkTree],
) -> None:
    """
    tests that a warning is logged if an unknown sampling strategy is provided.
    """
    # arrange
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_unknown_strategy", "test_dataset"
    raw_file_path = raw_dir / "results.json.gz"
    save_json_gz({"aspirin": [{}]}, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.adapt.return_value = iter(multiple_unique_trees)

    # act
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
        sampling_strategy="yeet-k",  # the key part of this test
        sample_k=1,
    )

    # assert
    assert "Unknown sampling strategy 'yeet-k'" in caplog.text
    # verify no sampling was actually applied
    manifest_path = processed_dir / generate_model_hash(model_name) / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    assert manifest["statistics"]["final_unique_routes_saved"] == len(multiple_unique_trees)
