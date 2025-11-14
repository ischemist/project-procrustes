from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from retrocast.adapters.common import build_tree_from_bipartite_node, build_tree_from_precursor_map
from retrocast.domain.chem import canonicalize_smiles
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
                    children=[
                        SimpleNamespace(smiles="CC=O", type="mol", in_stock=True, children=[]),  # acetaldehyde
                        SimpleNamespace(smiles="[H][H]", type="mol", in_stock=True, children=[]),  # hydrogen
                    ],
                )
            ],
        )

        tree = build_tree_from_bipartite_node(raw_data, "retrocast-mol-root")

        assert tree.smiles == "CCO"
        assert not tree.is_starting_material
        assert len(tree.reactions) == 1
        reaction = tree.reactions[0]
        assert len(reaction.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction.reactants}
        assert reactant_smiles == {"CC=O", "[H][H]"}
        assert all(r.is_starting_material for r in reaction.reactants)
        assert reaction.reaction_smiles == "CC=O.[H][H]>>CCO"

    def test_raises_on_malformed_input(self):
        """should fail if a molecule's child is not a reaction."""
        raw_data = SimpleNamespace(
            smiles="CCO",
            type="mol",
            in_stock=False,
            children=[SimpleNamespace(smiles="CC=O", type="mol", in_stock=True, children=[])],
        )
        with pytest.raises(AdapterLogicError, match="child of molecule node was not a reaction node"):
            build_tree_from_bipartite_node(raw_data, "retrocast-mol-root")

    def test_raises_on_invalid_tree_logic(self):
        """pydantic models should fail if a starting material has children."""
        raw_data = SimpleNamespace(
            smiles="CCO", type="mol", in_stock=True, children=[SimpleNamespace(type="reaction", children=[])]
        )
        # the builder will create it, but the pydantic validation on MoleculeNode will fail
        with pytest.raises(ValidationError, match="is a starting material but has 1 parent reactions"):
            build_tree_from_bipartite_node(raw_data, "retrocast-mol-root")


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
        tree = build_tree_from_precursor_map(root_smiles, precursor_map)

        assert tree.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert not tree.is_starting_material
        # first step
        reaction1 = tree.reactions[0]
        assert reaction1.reactants[0].smiles == "O=C(O)c1ccccc1O"
        assert not reaction1.reactants[0].is_starting_material
        # second step
        reaction2 = reaction1.reactants[0].reactions[0]
        assert reaction2.reactants[0].smiles == "Oc1ccccc1"
        assert reaction2.reactants[0].is_starting_material

    def test_build_convergent_route(self):
        """tests a route where a -> b + c."""
        precursor_map = {
            "A": ["B", "C"],
        }
        tree = build_tree_from_precursor_map("A", precursor_map)
        assert len(tree.reactions[0].reactants) == 2
        reactant_smiles = {r.smiles for r in tree.reactions[0].reactants}
        assert reactant_smiles == {"B", "C"}
        assert all(r.is_starting_material for r in tree.reactions[0].reactants)

    def test_handles_cycles(self):
        """tests that a cycle a -> b -> a is detected and handled."""
        precursor_map = {
            "A": ["B"],
            "B": ["A"],
        }
        tree = build_tree_from_precursor_map("A", precursor_map)
        # tree should be: A -> B -> (A as starting material)
        assert not tree.is_starting_material
        reactant_b = tree.reactions[0].reactants[0]
        assert reactant_b.smiles == "B"
        assert not reactant_b.is_starting_material
        reactant_a_cycle = reactant_b.reactions[0].reactants[0]
        assert reactant_a_cycle.smiles == "A"
        # cycle is broken, second A is a leaf
        assert reactant_a_cycle.is_starting_material
        assert not reactant_a_cycle.reactions
