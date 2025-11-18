import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.core import create_processing_manifest, process_model_run, process_raw_data
from retrocast.domain.chem import get_inchi_key
from retrocast.exceptions import RetroCastIOError
from retrocast.io import save_json_gz
from retrocast.schemas import Molecule, Route, RunStatistics, TargetInput
from retrocast.typing import SmilesStr
from retrocast.utils.hashing import generate_model_hash


@pytest.fixture
def aspirin_target_info() -> TargetInput:
    """Provides a standard TargetInput object for aspirin."""
    return TargetInput(
        id="aspirin",
        smiles=SmilesStr("CC(=O)OC1=CC=CC=C1C(=O)O"),
    )


@pytest.fixture
def minimal_fake_route(aspirin_target_info: TargetInput) -> Route:
    """Provides a minimal, valid Route for mocking adapter outputs."""
    target_molecule = Molecule(
        smiles=aspirin_target_info.smiles,
        inchikey=get_inchi_key(aspirin_target_info.smiles),
        synthesis_step=None,
    )
    return Route(target=target_molecule, rank=1)


@pytest.fixture
def multiple_unique_routes(aspirin_target_info: TargetInput) -> list[Route]:
    """Provides a list of three semantically unique Route objects with different structures."""
    from retrocast.schemas import ReactionStep

    # Route 1: aspirin as a leaf (no synthesis)
    route1 = Route(
        target=Molecule(
            smiles=aspirin_target_info.smiles,
            inchikey=get_inchi_key(aspirin_target_info.smiles),
            synthesis_step=None,
        ),
        rank=1,
    )

    # Route 2: aspirin from methanol + ethanol (different starting materials)
    route2 = Route(
        target=Molecule(
            smiles=aspirin_target_info.smiles,
            inchikey=get_inchi_key(aspirin_target_info.smiles),
            synthesis_step=ReactionStep(
                reactants=[
                    Molecule(smiles="CO", inchikey=get_inchi_key("CO"), synthesis_step=None),
                    Molecule(smiles="CCO", inchikey=get_inchi_key("CCO"), synthesis_step=None),
                ]
            ),
        ),
        rank=2,
    )

    # Route 3: aspirin from propanol + butanol (different starting materials)
    route3 = Route(
        target=Molecule(
            smiles=aspirin_target_info.smiles,
            inchikey=get_inchi_key(aspirin_target_info.smiles),
            synthesis_step=ReactionStep(
                reactants=[
                    Molecule(smiles="CCCO", inchikey=get_inchi_key("CCCO"), synthesis_step=None),
                    Molecule(smiles="CCCCO", inchikey=get_inchi_key("CCCCO"), synthesis_step=None),
                ]
            ),
        ),
        rank=3,
    )

    return [route1, route2, route3]


def test_process_model_run_no_sampling(
    tmp_path: Path, mocker: MockerFixture, aspirin_target_info: TargetInput, multiple_unique_routes: list[Route]
) -> None:
    """
    Tests the full orchestration without any sampling. Verifies correct output
    and that 'sampling_parameters' key is absent from the manifest.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_v1", "test_dataset"
    # Create raw data with 3 routes to match the 3 routes returned by adapter
    raw_file_content = {
        "aspirin": [
            {"smiles": "route1", "children": []},
            {"smiles": "route2", "children": []},
            {"smiles": "route3", "children": []},
        ]
    }
    raw_file_path = raw_dir / "target_aspirin.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes)

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

    # Verify all statistics are correct
    stats = manifest["statistics"]
    assert stats["total_routes_in_raw_files"] == 3  # 3 raw routes
    assert stats["final_unique_routes_saved"] == len(multiple_unique_routes)  # 3 routes
    # No failures since adapter yields all routes successfully
    assert stats["total_routes_failed_or_duplicate"] == 0
    assert stats["num_targets_with_at_least_one_route"] == 1
    assert stats["duplication_factor"] == 1.0  # No duplicates


@pytest.mark.parametrize(
    "strategy, k, saved_routes",
    [
        ("top-k", 2, 2),
        ("random-k", 1, 1),
        ("by-length", 2, 2),
    ],
)
def test_process_model_run_with_sampling_strategies(
    strategy: str,
    k: int,
    saved_routes: int,
    tmp_path: Path,
    mocker: MockerFixture,
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
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
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes)

    mock_sampling_func = mocker.MagicMock()
    mock_sampling_func.return_value = multiple_unique_routes[:saved_routes]
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
    mock_sampling_func.assert_called_once_with(multiple_unique_routes, k)

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
    tmp_path: Path, mocker: MockerFixture, caplog, aspirin_target_info: TargetInput
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
    mocker.patch("retrocast.core.load_json_gz", side_effect=RetroCastIOError("test error"))

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
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
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
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes)

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
    assert manifest["statistics"]["final_unique_routes_saved"] == len(multiple_unique_routes)


def test_process_model_run_warns_on_unknown_strategy(
    tmp_path: Path,
    mocker: MockerFixture,
    caplog,
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
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
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes)

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
    assert manifest["statistics"]["final_unique_routes_saved"] == len(multiple_unique_routes)


def test_process_model_run_tracks_failures(
    tmp_path: Path,
    mocker: MockerFixture,
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
) -> None:
    """
    Tests that statistics correctly track routes that fail during adapter processing.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_failures", "test_dataset"

    # Create raw data with 5 routes
    raw_file_content = {
        "aspirin": [
            {"smiles": "route1", "children": []},
            {"smiles": "route2", "children": []},
            {"smiles": "route3", "children": []},
            {"smiles": "route4", "children": []},
            {"smiles": "route5", "children": []},
        ]
    }
    raw_file_path = raw_dir / "results.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    # Mock adapter to only yield 2 successful routes (3 failures)
    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes[:2])

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
    manifest_path = processed_dir / generate_model_hash(model_name) / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    stats = manifest["statistics"]
    assert stats["total_routes_in_raw_files"] == 5  # 5 raw routes
    assert stats["final_unique_routes_saved"] == 2  # 2 successful routes
    # 3 routes failed during adapter processing (validation or transformation)
    assert stats["total_routes_failed_or_duplicate"] == 3
    assert stats["num_targets_with_at_least_one_route"] == 1


def test_process_model_run_tracks_duplicates(
    tmp_path: Path,
    mocker: MockerFixture,
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
) -> None:
    """
    Tests that statistics correctly track duplicate routes after deduplication.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "test_model_duplicates", "test_dataset"

    # Create raw data with 3 routes
    raw_file_content = {
        "aspirin": [
            {"smiles": "route1", "children": []},
            {"smiles": "route2", "children": []},
            {"smiles": "route3", "children": []},
        ]
    }
    raw_file_path = raw_dir / "results.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    # Mock adapter to yield route1, route1 (duplicate), route2
    # So 3 raw -> 3 transformed -> 2 unique
    duplicate_routes = [multiple_unique_routes[0], multiple_unique_routes[0], multiple_unique_routes[1]]
    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.cast.return_value = iter(duplicate_routes)

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
    manifest_path = processed_dir / generate_model_hash(model_name) / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    stats = manifest["statistics"]
    assert stats["total_routes_in_raw_files"] == 3  # 3 raw routes
    assert stats["final_unique_routes_saved"] == 2  # 2 unique routes after dedup
    # 0 routes failed processing, but 1 duplicate was removed
    assert stats["total_routes_failed_or_duplicate"] == 1
    assert stats["duplication_factor"] == 1.5  # 3 successful / 2 unique = 1.5


def test_process_raw_data_returns_correct_structure(
    tmp_path: Path,
    mocker: MockerFixture,
    aspirin_target_info: TargetInput,
    multiple_unique_routes: list[Route],
) -> None:
    """
    Tests that process_raw_data returns the expected tuple of stats and hashes.
    """
    # ARRANGE
    raw_data = {
        "aspirin": [
            {"smiles": "route1", "children": []},
            {"smiles": "route2", "children": []},
            {"smiles": "route3", "children": []},
        ]
    }
    mock_adapter = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter.cast.return_value = iter(multiple_unique_routes)

    output_path = tmp_path / "results.json.gz"

    # ACT
    stats, output_file_hash, routes_content_hash = process_raw_data(
        raw_data_per_target=raw_data,
        adapter=mock_adapter,
        targets_map={"aspirin": aspirin_target_info},
        output_path=output_path,
    )

    # ASSERT
    assert isinstance(stats, RunStatistics)
    assert isinstance(output_file_hash, str)
    assert isinstance(routes_content_hash, str)
    assert len(output_file_hash) == 64  # SHA256 hex digest length
    assert len(routes_content_hash) == 64  # SHA256 hex digest length
    assert output_path.exists()
    assert stats.total_routes_in_raw_files == 3
    assert stats.final_unique_routes_saved == 3


def test_create_processing_manifest_includes_all_fields(
    aspirin_target_info: TargetInput,
) -> None:
    """
    Tests that create_processing_manifest includes all expected fields.
    """
    # ARRANGE
    stats = RunStatistics()
    stats.total_routes_in_raw_files = 10
    stats.final_unique_routes_saved = 8
    stats.targets_with_at_least_one_route.add("aspirin")

    source_files = {"results.json.gz": "abc123"}
    output_file_hash = "def456" * 10  # Mock SHA256 hash
    routes_content_hash = "789ghi" * 10  # Mock content hash

    # ACT
    manifest = create_processing_manifest(
        model_name="test_model",
        dataset_name="test_dataset",
        source_files=source_files,
        output_file="results.json.gz",
        stats=stats,
        output_file_hash=output_file_hash,
        routes_content_hash=routes_content_hash,
        sampling_strategy="top-k",
        sample_k=5,
    )

    # ASSERT
    assert manifest["model_name"] == "test_model"
    assert manifest["dataset_name"] == "test_dataset"
    assert "output_files" in manifest
    assert "results.json.gz" in manifest["output_files"]
    assert manifest["output_files"]["results.json.gz"]["file_hash"] == output_file_hash
    assert manifest["output_files"]["results.json.gz"]["content_hash"] == routes_content_hash
    assert "model_hash" in manifest
    assert "source_hash" in manifest
    assert "processing_timestamp_utc" in manifest
    assert manifest["source_files"] == source_files
    assert manifest["sampling_parameters"]["strategy"] == "top-k"
    assert manifest["sampling_parameters"]["k"] == 5
    assert "statistics" in manifest


def test_create_processing_manifest_respects_custom_model_hash() -> None:
    """
    Tests that create_processing_manifest uses a custom model_hash when provided.
    """
    # ARRANGE
    stats = RunStatistics()
    custom_hash = "custom_model_hash_123"
    output_file_hash = ""
    routes_content_hash = ""

    # ACT
    manifest = create_processing_manifest(
        model_name="test_model",
        dataset_name="test_dataset",
        source_files={},
        output_file=None,
        stats=stats,
        output_file_hash=output_file_hash,
        routes_content_hash=routes_content_hash,
        model_hash=custom_hash,
    )

    # ASSERT
    assert manifest["model_hash"] == custom_hash


def test_process_model_run_with_anonymize_false(
    tmp_path: Path, mocker: MockerFixture, aspirin_target_info: TargetInput, multiple_unique_routes: list[Route]
) -> None:
    """
    Tests that when anonymize=False, the output directory uses the model name directly.
    """
    # ARRANGE
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), processed_dir.mkdir()
    model_name, dataset_name = "my-test-model", "test_dataset"
    raw_file_content = {"aspirin": [{"smiles": "...", "children": []}]}
    raw_file_path = raw_dir / "results.json.gz"
    save_json_gz(raw_file_content, raw_file_path)

    mock_adapter_instance = mocker.MagicMock(spec=BaseAdapter)
    mock_adapter_instance.cast.return_value = iter(multiple_unique_routes)

    # ACT
    process_model_run(
        model_name=model_name,
        dataset_name=dataset_name,
        adapter=mock_adapter_instance,
        raw_results_file=raw_file_path,
        processed_dir=processed_dir,
        targets_map={"aspirin": aspirin_target_info},
        anonymize=False,  # Key part of this test
    )

    # ASSERT
    # Output should be in model_name directory, not hashed
    output_dir = processed_dir / model_name
    assert output_dir.exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "results.json.gz").exists()

    # Hashed directory should NOT exist
    hashed_dir = processed_dir / generate_model_hash(model_name)
    assert not hashed_dir.exists()

    # Manifest should still contain the hash
    with (output_dir / "manifest.json").open("r") as f:
        manifest = json.load(f)
    assert manifest["model_name"] == model_name
    assert manifest["model_hash"] == generate_model_hash(model_name)
