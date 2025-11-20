import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import Molecule, ReactionStep, Route, TargetIdentity, TargetInput
from retrocast.typing import InchiKeyStr, SmilesStr
from tests.helpers import _make_leaf_molecule, _synthetic_inchikey

TEST_DATA_DIR = Path("tests/testing_data")
MODEL_PRED_DIR = TEST_DATA_DIR / "model-predictions"


# =============================================================================
# Synthetic Route Factory for Topology Testing
# =============================================================================


def _carbon_chain_smiles(n: int) -> str:
    """Generate SMILES for a carbon chain of length n. C, CC, CCC, etc."""
    if n <= 0:
        raise ValueError("Carbon chain length must be positive")
    return "C" * n


def _make_linear_route(depth: int) -> Route:
    """
    Create a linear route with the given depth.

    Depth 1: CC <- C (one reaction)
    Depth 2: CCC <- CC <- C (two reactions)
    Depth 3: CCCC <- CCC <- CC <- C (three reactions)
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    # Start with the leaf (single carbon)
    current = _make_leaf_molecule(_carbon_chain_smiles(1))

    # Build up the chain
    for i in range(2, depth + 2):
        product_smiles = _carbon_chain_smiles(i)
        current = Molecule(
            smiles=SmilesStr(product_smiles),
            inchikey=InchiKeyStr(_synthetic_inchikey(product_smiles)),
            synthesis_step=ReactionStep(reactants=[current]),
        )

    return Route(target=current, rank=1)


def _make_convergent_route(depth: int) -> Route:
    """
    Create a convergent route where two branches merge at the top.
    
    Depth 2:
        CCCC
        /  \
       CC   CC
       |    |
       C    C
    
    Each branch is linear with depth-1 steps, then they merge.
    """
    if depth < 2:
        raise ValueError("Convergent routes require depth >= 2")

    # Build two independent branches
    branch_depth = depth - 1

    # Branch 1: C -> CC -> ...
    branch1 = _make_leaf_molecule(_carbon_chain_smiles(1))
    for i in range(2, branch_depth + 2):
        branch1_smiles = _carbon_chain_smiles(i)
        branch1 = Molecule(
            smiles=SmilesStr(branch1_smiles),
            inchikey=InchiKeyStr(_synthetic_inchikey(f"branch1_{branch1_smiles}")),
            synthesis_step=ReactionStep(reactants=[branch1]),
        )

    # Branch 2: C -> CC -> ... (different inchikeys via prefix)
    branch2 = _make_leaf_molecule("O")  # Use oxygen as second leaf for variety
    for i in range(2, branch_depth + 2):
        branch2_smiles = _carbon_chain_smiles(i)
        branch2 = Molecule(
            smiles=SmilesStr(branch2_smiles),
            inchikey=InchiKeyStr(_synthetic_inchikey(f"branch2_{branch2_smiles}")),
            synthesis_step=ReactionStep(reactants=[branch2]),
        )

    # Merge the two branches
    final_smiles = _carbon_chain_smiles(depth + 2)
    final = Molecule(
        smiles=SmilesStr(final_smiles),
        inchikey=InchiKeyStr(_synthetic_inchikey(final_smiles)),
        synthesis_step=ReactionStep(reactants=[branch1, branch2]),
    )

    return Route(target=final, rank=1)


def _make_binary_tree_route(depth: int) -> Route:
    """
    Create a fully convergent binary tree route.

    Depth 1: CC <- (C + C)
    Depth 2: CCCC <- (CC <- (C + C)) + (CC <- (C + C))

    At each level, two subtrees merge.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    leaf_counter = [0]  # Mutable counter for unique leaves

    def _build_tree(current_depth: int) -> Molecule:
        if current_depth == 0:
            # Create unique leaf (all have same SMILES but unique InchiKeys)
            leaf_counter[0] += 1
            return Molecule(
                smiles=SmilesStr("C"),
                inchikey=InchiKeyStr(_synthetic_inchikey(f"leaf_{leaf_counter[0]}")),
                synthesis_step=None,
            )

        # Build two subtrees and merge
        left = _build_tree(current_depth - 1)
        right = _build_tree(current_depth - 1)

        product_smiles = _carbon_chain_smiles(2**current_depth)
        return Molecule(
            smiles=SmilesStr(product_smiles),
            inchikey=InchiKeyStr(_synthetic_inchikey(f"node_{current_depth}_{leaf_counter[0]}")),
            synthesis_step=ReactionStep(reactants=[left, right]),
        )

    return Route(target=_build_tree(depth), rank=1)


@pytest.fixture
def synthetic_route_factory():
    """
    Factory fixture for creating synthetic routes using carbon chains.

    This enables testing route topology (depth, convergence, hashing) without
    chemical complexity. Uses deterministic fake InchiKeys for reproducibility.

    Args:
        structure: "linear", "convergent", or "binary_tree"
        depth: Number of reaction steps (minimum 1, convergent requires >= 2)

    Returns:
        A Route object with the specified topology.

    Examples:
        # Linear route with 3 steps: CCCC <- CCC <- CC <- C
        route = synthetic_route_factory("linear", depth=3)

        # Convergent route: two branches merge at top
        route = synthetic_route_factory("convergent", depth=3)

        # Binary tree: fully convergent at every level
        route = synthetic_route_factory("binary_tree", depth=2)
    """

    def _make(structure: str = "linear", depth: int = 3) -> Route:
        if structure == "linear":
            return _make_linear_route(depth)
        elif structure == "convergent":
            return _make_convergent_route(depth)
        elif structure == "binary_tree":
            return _make_binary_tree_route(depth)
        else:
            raise ValueError(f"Unknown structure: {structure}. Use 'linear', 'convergent', or 'binary_tree'")

    return _make


@pytest.fixture
def leaf_molecule_factory():
    """Factory for creating leaf molecules with synthetic InchiKeys."""

    def _make(smiles: str = "C") -> Molecule:
        return _make_leaf_molecule(smiles)

    return _make


@pytest.fixture
def synthetic_stock() -> set[str]:
    """A minimal stock containing common synthetic leaves."""
    return {
        _synthetic_inchikey("C"),
        _synthetic_inchikey("O"),
    }


@pytest.fixture(scope="session")
def multistepttl_ibuprofen_dir() -> Path:
    """provides the path to the directory containing ibuprofen pickles for multistepttl."""
    return Path(MODEL_PRED_DIR / "multistepttl/ibuprofen_multistepttl")


@pytest.fixture(scope="session")
def multistepttl_paracetamol_dir() -> Path:
    """provides the path to the directory containing paracetamol pickles for multistepttl."""
    return Path(MODEL_PRED_DIR / "multistepttl/paracetamol_multistepttl")


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
