import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from ursa.domain.chem import canonicalize_smiles
from ursa.domain.schemas import TargetInfo

MODEL_PRED_DIR = Path("tests/testing_data/model-predictions")


@pytest.fixture(scope="session")
def raw_aizynth_mcts_data() -> dict[str, Any]:
    """loads the raw aizynthfinder mcts prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "aizynthfinder-mcts/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_aizynth_retro_star_data() -> dict[str, Any]:
    """loads the raw aizynthfinder retro-star prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "aizynthfinder-retro-star/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_askcos_data() -> dict[str, Any]:
    """loads the raw askcos prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "askcos/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_retrostar_data() -> dict[str, Any]:
    """loads the raw retro-star prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "retro-star/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_dms_data() -> dict[str, Any]:
    """loads the raw dms prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "dms-flash-fp16/ursa_bb_results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_retrochimera_data() -> dict[str, Any]:
    """loads the raw retrochimera prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "retrochimera/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_dreamretro_data() -> dict[str, Any]:
    """loads the raw dreamretro prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "dreamretro/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def multistepttl_ibuprofen_dir() -> Path:
    """provides the path to the directory containing ibuprofen pickles for multistepttl."""
    return Path(MODEL_PRED_DIR / "multistepttl/ibuprofen_multistepttl")


@pytest.fixture(scope="session")
def multistepttl_paracetamol_dir() -> Path:
    """provides the path to the directory containing paracetamol pickles for multistepttl."""
    return Path(MODEL_PRED_DIR / "multistepttl/paracetamol_multistepttl")


@pytest.fixture(scope="session")
def raw_synplanner_data() -> dict[str, Any]:
    """loads the raw synplanner prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "synplanner-mcts/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_syntheseus_data() -> dict[str, Any]:
    """loads the raw syntheseus prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "syntheseus-retro0-local-retro/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def methylacetate_target_info() -> TargetInfo:
    """provides the target info object for methyl acetate."""
    return TargetInfo(id="methylacetate", smiles=canonicalize_smiles("COC(C)=O"))
