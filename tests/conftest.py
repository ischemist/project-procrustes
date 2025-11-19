import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetIdentity, TargetInput

TEST_DATA_DIR = Path("tests/testing_data")
MODEL_PRED_DIR = TEST_DATA_DIR / "model-predictions"


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
def raw_synllama_data() -> dict[str, Any]:
    """loads the raw syntheseus prediction data from the test file."""
    path = Path(MODEL_PRED_DIR / "synllama/results.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_paroutes_data() -> dict[str, Any]:
    """loads the raw syntheseus prediction data from the test file."""
    path = Path(TEST_DATA_DIR / "paroutes.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pharma_routes_data() -> dict[str, Any]:
    """loads the pharma routes data from the test file for contract/regression tests."""
    path = Path(TEST_DATA_DIR / "pharma_routes.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def methylacetate_target_input() -> TargetIdentity:
    """provides the target input object for methyl acetate."""
    return TargetInput(id="methylacetate", smiles=canonicalize_smiles("COC(C)=O"))


@pytest.fixture(scope="session")
def sample_routes_with_reactions() -> dict[str, list]:
    """
    Creates a small set of routes with actual reaction steps for testing curation functions.

    Structure:
    - target_A: 2 routes for ethyl acetate synthesis
      - Route 1: EtOAc <- (EtOH + AcOH)
      - Route 2: EtOAc <- (EtOH + Ac2O)
    - target_B: 1 route for aspirin synthesis
      - Route 1: Aspirin <- (Salicylic acid + Ac2O) where Salicylic acid <- (Phenol + CO2)
    """
    from retrocast.models.chem import Molecule, ReactionStep, Route
    from retrocast.typing import InchiKeyStr, SmilesStr

    # Define leaf molecules (building blocks)
    ethanol = Molecule(smiles=SmilesStr("CCO"), inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"))
    acetic_acid = Molecule(smiles=SmilesStr("CC(=O)O"), inchikey=InchiKeyStr("QTBSBXVTEAMEQO-UHFFFAOYSA-N"))
    acetic_anhydride = Molecule(smiles=SmilesStr("CC(=O)OC(C)=O"), inchikey=InchiKeyStr("WFDIJRYMOXRFFG-UHFFFAOYSA-N"))
    phenol = Molecule(smiles=SmilesStr("Oc1ccccc1"), inchikey=InchiKeyStr("ISWSIDIOOBJBQZ-UHFFFAOYSA-N"))
    co2 = Molecule(smiles=SmilesStr("O=C=O"), inchikey=InchiKeyStr("CURLTUGMZLYLDI-UHFFFAOYSA-N"))

    # Route 1 for target_A: EtOAc from EtOH + AcOH
    ethyl_acetate_1 = Molecule(
        smiles=SmilesStr("CCOC(C)=O"),
        inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
        synthesis_step=ReactionStep(reactants=[ethanol, acetic_acid]),
    )
    route_A1 = Route(target=ethyl_acetate_1, rank=1)

    # Route 2 for target_A: EtOAc from EtOH + Ac2O
    ethyl_acetate_2 = Molecule(
        smiles=SmilesStr("CCOC(C)=O"),
        inchikey=InchiKeyStr("XEKOWRVHYACXOJ-UHFFFAOYSA-N"),
        synthesis_step=ReactionStep(reactants=[ethanol, acetic_anhydride]),
    )
    route_A2 = Route(target=ethyl_acetate_2, rank=2)

    # Route for target_B: Aspirin synthesis (2-step)
    # Step 1: Salicylic acid from phenol + CO2
    salicylic_acid = Molecule(
        smiles=SmilesStr("O=C(O)c1ccccc1O"),
        inchikey=InchiKeyStr("YGSDEFSMJLZEOE-UHFFFAOYSA-N"),
        synthesis_step=ReactionStep(reactants=[phenol, co2]),
    )
    # Step 2: Aspirin from salicylic acid + Ac2O
    aspirin = Molecule(
        smiles=SmilesStr("CC(=O)Oc1ccccc1C(=O)O"),
        inchikey=InchiKeyStr("BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
        synthesis_step=ReactionStep(reactants=[salicylic_acid, acetic_anhydride]),
    )
    route_B1 = Route(target=aspirin, rank=1)

    return {"target_A": [route_A1, route_A2], "target_B": [route_B1]}
