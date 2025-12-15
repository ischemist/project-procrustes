import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pytest

TEST_DATA_DIR = Path("tests/testing_data")
MODEL_PRED_DIR = TEST_DATA_DIR / "model-predictions"


@pytest.fixture(autouse=True)
def configure_adapter_logging(caplog):
    """Ensure all adapter loggers capture DEBUG logs in tests."""
    caplog.set_level(logging.DEBUG, logger="retrocast.adapters")


@pytest.fixture(scope="session")
def raw_data_factory():
    """Factory to load gzipped JSON data from test files."""

    def _load_data(relative_path: str | Path) -> dict[str, Any]:
        """Load gzipped JSON data from a path relative to tests/testing_data.

        Args:
            relative_path: Path relative to TEST_DATA_DIR

        Returns:
            Loaded JSON data as a dictionary
        """
        path = TEST_DATA_DIR / relative_path
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    return _load_data


@pytest.fixture(scope="session")
def raw_aizynth_mcts_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw aizynthfinder mcts prediction data from the test file."""
    return raw_data_factory("model-predictions/aizynthfinder-mcts/results.json.gz")


@pytest.fixture(scope="session")
def raw_aizynth_retro_star_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw aizynthfinder retro-star prediction data from the test file."""
    return raw_data_factory("model-predictions/aizynthfinder-retro-star/results.json.gz")


@pytest.fixture(scope="session")
def raw_askcos_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw askcos prediction data from the test file."""
    return raw_data_factory("model-predictions/askcos/results.json.gz")


@pytest.fixture(scope="session")
def raw_retrostar_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw retro-star prediction data from the test file."""
    return raw_data_factory("model-predictions/retro-star/results.json.gz")


@pytest.fixture(scope="session")
def raw_dms_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw dms prediction data from the test file."""
    return raw_data_factory("model-predictions/dms-flash-fp16/ursa_bb_results.json.gz")


@pytest.fixture(scope="session")
def raw_retrochimera_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw retrochimera prediction data from the test file."""
    return raw_data_factory("model-predictions/retrochimera/results.json.gz")


@pytest.fixture(scope="session")
def raw_dreamretro_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw dreamretro prediction data from the test file."""
    return raw_data_factory("model-predictions/dreamretro/results.json.gz")


@pytest.fixture(scope="session")
def raw_synplanner_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw synplanner prediction data from the test file."""
    return raw_data_factory("model-predictions/synplanner-mcts-rollout/results.json.gz")


@pytest.fixture(scope="session")
def raw_syntheseus_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw syntheseus prediction data from the test file."""
    return raw_data_factory("model-predictions/syntheseus-retro0-local-retro/results.json.gz")


@pytest.fixture(scope="session")
def raw_synllama_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw synllama prediction data from the test file."""
    return raw_data_factory("model-predictions/synllama/results.json.gz")


@pytest.fixture(scope="session")
def raw_paroutes_data(raw_data_factory) -> dict[str, Any]:
    """loads the raw paroutes prediction data from the test file."""
    return raw_data_factory("paroutes.json.gz")
