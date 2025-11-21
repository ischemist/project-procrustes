import gzip
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from retrocast.utils.serializers import (
    SyntheseusSerializationError,
    TtlRetroSerializationError,
    serialize_and_save,
    serialize_multistepttl_directory,
    serialize_multistepttl_target,
    serialize_route,
)

# --- Mock Objects to simulate syntheseus classes ---


def mock_mol(smiles: str, is_purchasable: bool = False) -> SimpleNamespace:
    return SimpleNamespace(smiles=smiles, metadata={"is_purchasable": is_purchasable})


def mock_rxn(reactants: tuple[SimpleNamespace, ...], product: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(reactants=reactants, products=(product,))


def mock_or_node(smiles: str, is_purchasable: bool = False) -> SimpleNamespace:
    return SimpleNamespace(mol=mock_mol(smiles, is_purchasable))


def mock_and_node(reactants: tuple[str, ...], product: str, is_purchasable: bool = False) -> SimpleNamespace:
    reactant_mols = tuple(mock_mol(s) for s in reactants)
    product_mol = mock_mol(product, is_purchasable)
    return SimpleNamespace(reaction=mock_rxn(reactant_mols, product_mol))


@pytest.fixture
def target_smiles() -> str:
    return "Cc1ccc(-c2ccc(C)cc2)cc1"


@pytest.fixture
def simple_route(target_smiles) -> list:
    """A -> B + C; B is purchasable, C -> D; D is purchasable."""
    mol_A = mock_or_node(target_smiles)
    mol_B = mock_or_node("Cc1ccc(B(O)O)cc1", is_purchasable=True)
    mol_C = mock_or_node("Cc1ccc(I)cc1")
    mol_D = mock_or_node("O=Cc1ccc(I)cc1", is_purchasable=True)

    rxn_A = mock_and_node(reactants=("Cc1ccc(B(O)O)cc1", "Cc1ccc(I)cc1"), product=target_smiles)
    rxn_C = mock_and_node(reactants=("O=Cc1ccc(I)cc1",), product="Cc1ccc(I)cc1")

    return [mol_A, mol_B, mol_C, mol_D, rxn_A, rxn_C]


@pytest.fixture
def purchasable_target_route() -> list:
    """A route where the target itself is purchasable."""
    return [mock_or_node("CCO", is_purchasable=True)]


@pytest.mark.unit
class TestSyntheseusSerializer:
    def test_serialize_route_simple(self, simple_route, target_smiles):
        """Tests that a valid, multi-step route is serialized correctly."""
        result = serialize_route(simple_route, target_smiles)

        assert result["smiles"] == target_smiles
        assert not result["in_stock"]
        reaction_a = result["children"][0]
        assert len(reaction_a["children"]) == 2
        sorted_reactants = sorted(reaction_a["children"], key=lambda x: x["smiles"])
        assert sorted_reactants[0]["smiles"] == "Cc1ccc(B(O)O)cc1"
        assert sorted_reactants[0]["in_stock"]
        assert sorted_reactants[1]["smiles"] == "Cc1ccc(I)cc1"
        assert not sorted_reactants[1]["in_stock"]

    def test_serialize_route_purchasable_target(self, purchasable_target_route):
        """Tests the base case where the target itself is a leaf node."""
        target_smiles = "CCO"
        result = serialize_route(purchasable_target_route, target_smiles)
        assert result["smiles"] == target_smiles
        assert result["in_stock"]
        assert len(result["children"]) == 0

    def test_serialize_route_missing_target_smiles_raises_error(self, simple_route):
        """Tests that an error is raised if the target SMILES isn't in the graph."""
        with pytest.raises(SyntheseusSerializationError, match="not found in the provided route nodes"):
            serialize_route(simple_route, "invalid_smiles")

    def test_serialize_route_incomplete_graph_raises_error(self, simple_route, target_smiles):
        """Tests that an error is raised if a reactant's OrNode is missing."""
        incomplete_route = [
            node for node in simple_route if not (hasattr(node, "mol") and node.mol.smiles == "Cc1ccc(I)cc1")
        ]
        expected_error_msg = re.escape("Incomplete route graph: OrNode for SMILES 'Cc1ccc(I)cc1' not found.")
        with pytest.raises(SyntheseusSerializationError, match=expected_error_msg):
            serialize_route(incomplete_route, target_smiles)

    def test_serialize_and_save_happy_path(self, tmp_path: Path, simple_route, target_smiles, purchasable_target_route):
        """
        Tests the main entry point, ensuring it handles multiple targets and writes
        the correct gzipped JSON file.
        """
        routes_by_target = {
            target_smiles: [simple_route],
            "CCO": [purchasable_target_route],
        }
        output_path = tmp_path / "syntheseus_output.json.gz"

        serialize_and_save(routes_by_target, output_path)

        assert output_path.exists()
        with gzip.open(output_path, "rt") as f:
            data = json.load(f)

        assert len(data) == 2
        assert "CCO" in data
        assert target_smiles in data
        assert data["CCO"][0]["smiles"] == "CCO"
        assert data[target_smiles][0]["smiles"] == target_smiles

    def test_serialize_and_save_handles_serialization_error(self, tmp_path: Path, simple_route, target_smiles, capsys):
        """
        Tests that if one route fails serialization, a warning is printed, but
        other successful routes are still written to the output file.
        """
        # This route is valid, but we'll add a second, broken one that is missing
        # an OrNode for one of its reactants, which will cause an error.
        broken_target_smiles = "BROKEN_TARGET"
        target_node = mock_or_node(broken_target_smiles)
        reaction_node = mock_and_node(reactants=("MISSING_REACTANT",), product=broken_target_smiles)
        broken_route_nodes = [target_node, reaction_node]
        routes_by_target = {
            target_smiles: [simple_route],
            broken_target_smiles: [broken_route_nodes],
        }
        output_path = tmp_path / "syntheseus_output_partial.json.gz"
        serialize_and_save(routes_by_target, output_path)

        # Check that the warning was printed to stdout
        captured = capsys.readouterr()
        assert f"Warning: Could not serialize route 0 for target {broken_target_smiles}" in captured.out

        # Check that the output file contains the valid route
        with gzip.open(output_path, "rt") as f:
            data = json.load(f)
        assert len(data) == 2
        assert target_smiles in data
        assert data[target_smiles][0]["smiles"] == target_smiles
        assert broken_target_smiles in data
        assert data[broken_target_smiles] == []  # The list for the failed target is empty


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:numpy.core.numeric is deprecated:DeprecationWarning")
class TestTtlRetroSerializer:
    def test_serialize_ibuprofen_directory(self, multistepttl_ibuprofen_dir: Path):
        """tests end-to-end serialization from a directory for a multi-step route."""
        serialized_routes = serialize_multistepttl_directory(multistepttl_ibuprofen_dir)

        assert isinstance(serialized_routes, list)
        assert len(serialized_routes) == 13

        first_route = serialized_routes[0]
        assert "reactions" in first_route
        assert first_route["metadata"]["steps"] == 2
        assert serialized_routes[10]["metadata"]["steps"] == 3

    def test_serialize_paracetamol_directory(self, multistepttl_paracetamol_dir: Path):
        """tests end-to-end serialization for a single-step route."""
        serialized_routes = serialize_multistepttl_directory(multistepttl_paracetamol_dir)

        assert isinstance(serialized_routes, list)
        assert len(serialized_routes) == 23

        first_route = serialized_routes[0]
        assert len(first_route["reactions"]) == 1
        assert first_route["metadata"]["steps"] == 1

    def test_directory_not_found(self, tmp_path: Path):
        """tests that a directory without pickles returns None."""
        non_existent_dir = tmp_path / "ghost"
        non_existent_dir.mkdir()
        assert serialize_multistepttl_directory(non_existent_dir) is None

    def test_no_solved_routes(self):
        """tests serialization when the tree_df has no solved routes."""
        tree_df = pd.DataFrame({"Solved": ["No", "No"], "Route": [[1], [2]], "Score": [0.5, 0.4]})
        predictions_df = pd.DataFrame(index=[1, 2])
        predictions_df["index"] = predictions_df.index

        result = serialize_multistepttl_target(tree_df, predictions_df)
        assert result == []

    def test_inconsistent_data_raises_error(self):
        """tests that a missing reaction id in predictions raises an error."""
        tree_df = pd.DataFrame({"Solved": ["Yes"], "Route": [[1, 99]], "Score": [0.5]})
        predictions_df = pd.DataFrame({"Prob_Forward_Prediction_1": [0.9]}, index=[1])
        predictions_df["index"] = predictions_df.index

        with pytest.raises(TtlRetroSerializationError, match="reaction id 99 from route not found"):
            serialize_multistepttl_target(tree_df, predictions_df)

    def test_serialize_multistepttl_directory_raises_on_processing_error(self, tmp_path: Path, mocker: MockerFixture):
        """Ensures that generic exceptions during pickle processing are wrapped."""
        target_dir = tmp_path / "bad_pickles"
        target_dir.mkdir()
        (target_dir / "mock__tree.pkl").touch()
        (target_dir / "mock__prediction.pkl").touch()

        # Mock pd.read_pickle to raise a generic error
        mocker.patch("pandas.read_pickle", side_effect=ValueError("Corrupt pickle"))

        with pytest.raises(TtlRetroSerializationError, match="failed to process pickles"):
            serialize_multistepttl_directory(target_dir)
