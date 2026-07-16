from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent / "testing_data"
MODEL_PRED_DIR = TEST_DATA_DIR / "model-predictions"


@pytest.fixture(scope="session")
def multistepttl_ibuprofen_dir() -> Path:
    return MODEL_PRED_DIR / "multistepttl/ibuprofen_multistepttl"


@pytest.fixture(scope="session")
def multistepttl_paracetamol_dir() -> Path:
    return MODEL_PRED_DIR / "multistepttl/paracetamol_multistepttl"
