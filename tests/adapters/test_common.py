from types import SimpleNamespace

import pytest

from retrocast.adapters.common import build_molecule_from_bipartite_node, build_molecule_from_precursor_map
from retrocast.chem import canonicalize_smiles
from retrocast.exceptions import AdapterLogicError


class TestBipartiteBuilder:
    def test_build_simple_one_step(self):
        """tests a simple a -> b + c conversion."""
        raw_data = SimpleNamespace(
            smiles="CCO",  # ethanol
            type="mol",
            in_stock=False,
            children=[
                SimpleNamespace(
                    type="reaction",
                    metadata={},
                    children=[
                        SimpleNamespace(smiles="CC=O", type="mol", in_stock=True, children=[]),  # acetaldehyde
                        SimpleNamespace(smiles="[H][H]", type="mol", in_stock=True, children=[]),  # hydrogen
                    ],
                )
            ],
        )

        molecule = build_molecule_from_bipartite_node(raw_data)

        assert molecule.smiles == "CCO"
        assert not molecule.is_leaf
        assert molecule.synthesis_step is not None
        reaction = molecule.synthesis_step
        assert len(reaction.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction.reactants}
        assert reactant_smiles == {"CC=O", "[H][H]"}
        assert all(r.is_leaf for r in reaction.reactants)

    def test_raises_on_malformed_input(self):
        """should fail if a molecule's child is not a reaction."""
        raw_data = SimpleNamespace(
            smiles="CCO",
            type="mol",
            in_stock=False,
            children=[SimpleNamespace(smiles="CC=O", type="mol", in_stock=True, children=[])],
        )
        with pytest.raises(AdapterLogicError) as exc_info:
            build_molecule_from_bipartite_node(raw_data)
        assert exc_info.value.code == "adapter.node_type_invalid"
        assert exc_info.value.context["expected"] == "reaction"

    def test_raises_on_cycle(self):
        """should fail if the same molecule is revisited in one path."""
        raw_data = SimpleNamespace(
            smiles="CCO",
            type="mol",
            in_stock=False,
            children=[
                SimpleNamespace(
                    type="reaction",
                    metadata={},
                    children=[
                        SimpleNamespace(
                            smiles="C",
                            type="mol",
                            in_stock=False,
                            children=[
                                SimpleNamespace(
                                    type="reaction",
                                    metadata={},
                                    children=[
                                        SimpleNamespace(
                                            smiles="CCO",
                                            type="mol",
                                            in_stock=False,
                                            children=[],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        with pytest.raises(AdapterLogicError) as exc_info:
            build_molecule_from_bipartite_node(raw_data)
        assert exc_info.value.code == "adapter.cycle_detected"


class TestPrecursorMapBuilder:
    def test_build_linear_route(self):
        """tests a simple a -> b -> c conversion."""
        precursor_map = {
            canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O"): [
                canonicalize_smiles("O=C(O)c1ccccc1O")
            ],  # aspirin -> salicylic
            canonicalize_smiles("O=C(O)c1ccccc1O"): [canonicalize_smiles("c1ccccc1O")],  # salicylic -> phenol
        }
        root_smiles = canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        molecule = build_molecule_from_precursor_map(root_smiles, precursor_map)

        assert molecule.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert not molecule.is_leaf
        # first step
        reaction1 = molecule.synthesis_step
        assert reaction1 is not None
        assert reaction1.reactants[0].smiles == "O=C(O)c1ccccc1O"
        assert not reaction1.reactants[0].is_leaf
        # second step
        reaction2 = reaction1.reactants[0].synthesis_step
        assert reaction2 is not None
        assert reaction2.reactants[0].smiles == "Oc1ccccc1"
        assert reaction2.reactants[0].is_leaf

    def test_build_convergent_route(self):
        """tests a route where a -> b + c."""
        precursor_map = {
            "CCO": ["C", "CO"],  # ethanol <- methane + methanol (simplified)
        }
        molecule = build_molecule_from_precursor_map("CCO", precursor_map)
        assert molecule.synthesis_step is not None
        assert len(molecule.synthesis_step.reactants) == 2
        reactant_smiles = {r.smiles for r in molecule.synthesis_step.reactants}
        assert reactant_smiles == {"C", "CO"}
        assert all(r.is_leaf for r in molecule.synthesis_step.reactants)

    def test_raises_on_cycles(self):
        """tests that a cycle a -> b -> a raises a typed adapter error."""
        precursor_map = {
            "CCO": ["C"],  # ethanol from methane
            "C": ["CCO"],  # methane from ethanol (cycle)
        }
        with pytest.raises(AdapterLogicError) as exc_info:
            build_molecule_from_precursor_map("CCO", precursor_map)
        assert exc_info.value.code == "adapter.cycle_detected"

    def test_raises_on_cycles_when_aliases_share_a_canonical_smiles(self):
        """different smiles spellings of the same molecule should still trip cycle detection."""
        precursor_map = {
            "CCO": ["OCC"],
            "OCC": ["C"],
            "C": ["CCO"],
        }

        with pytest.raises(AdapterLogicError) as exc_info:
            build_molecule_from_precursor_map("CCO", precursor_map)

        assert exc_info.value.code == "adapter.cycle_detected"
        assert exc_info.value.context["smiles"] == canonicalize_smiles("CCO")
