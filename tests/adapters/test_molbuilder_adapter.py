"""Tests for the MolBuilder adapter.

Unit tests inherit from BaseAdapterTest to verify standard adapter contracts.
Additional tests cover MolBuilder-specific features: reaction metadata
propagation, multi-step trees, cycle detection, and purchasability handling.
"""

import logging

import pytest

from retrocast.adapters.molbuilder_adapter import MolBuilderAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.models.chem import TargetInput
from tests.adapters.test_base_adapter import BaseAdapterTest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
#  Unit tests (inherit base adapter contract)
# ---------------------------------------------------------------------------


class TestMolBuilderAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return MolBuilderAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        """One-step route: ethanol from acetaldehyde + hydrogen."""
        return [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {
                    "reaction_name": "Reduction",
                    "named_reaction": "NaBH4 Reduction",
                    "category": "reduction",
                    "score": 0.85,
                    "precursors": [
                        {"smiles": "CC=O", "name": "acetaldehyde", "cost_per_kg": 15.0},
                        {"smiles": "[H][H]", "name": "hydrogen", "cost_per_kg": 5.0},
                    ],
                },
                "children": [
                    {"smiles": "CC=O", "is_purchasable": True, "children": []},
                    {"smiles": "[H][H]", "is_purchasable": True, "children": []},
                ],
            }
        ]

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        """Empty list means no routes found."""
        return []

    @pytest.fixture
    def raw_invalid_schema_data(self):
        """'children' is a string instead of list -- fails Pydantic validation."""
        return [{"smiles": "CCO", "children": "not a list"}]

    @pytest.fixture
    def target_input(self):
        return TargetInput(id="ethanol", smiles="CCO")

    @pytest.fixture
    def mismatched_target_input(self):
        return TargetInput(id="ethanol", smiles="CCC")

    # --- MolBuilder-specific tests ---

    def test_reaction_metadata_propagated(self, adapter_instance, raw_valid_route_data, target_input):
        """Verify reaction name and score are preserved in step metadata."""
        routes = list(adapter_instance.cast(raw_valid_route_data, target_input))
        assert len(routes) == 1
        route = routes[0]

        step = route.target.synthesis_step
        assert step is not None
        assert step.metadata["reaction_name"] == "Reduction"
        assert step.metadata["named_reaction"] == "NaBH4 Reduction"
        assert step.metadata["category"] == "reduction"
        assert step.metadata["score"] == 0.85

    def test_template_set_to_reaction_name(self, adapter_instance, raw_valid_route_data, target_input):
        """MolBuilder uses name-based templates, stored in ReactionStep.template."""
        routes = list(adapter_instance.cast(raw_valid_route_data, target_input))
        step = routes[0].target.synthesis_step
        assert step is not None
        assert step.template == "Reduction"

    def test_route_metadata_has_score(self, adapter_instance, raw_valid_route_data, target_input):
        """Score from root disconnection is copied to route-level metadata."""
        routes = list(adapter_instance.cast(raw_valid_route_data, target_input))
        assert routes[0].metadata["score"] == 0.85

    def test_purchasable_nodes_are_leaves(self, adapter_instance, target_input):
        """Nodes with is_purchasable=True should be leaves even if they have children."""
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {
                    "reaction_name": "Test",
                    "score": 0.5,
                },
                "children": [
                    {
                        "smiles": "CC=O",
                        "is_purchasable": True,
                        # Purchasable with children -- should still be a leaf
                        "children": [
                            {"smiles": "C", "is_purchasable": True, "children": []},
                        ],
                    },
                ],
            }
        ]
        routes = list(adapter_instance.cast(raw_routes, target_input))
        assert len(routes) == 1
        step = routes[0].target.synthesis_step
        assert step is not None
        assert len(step.reactants) == 1
        assert step.reactants[0].is_leaf  # purchasable -> leaf

    def test_multiple_routes_ranked(self, adapter_instance):
        """Multiple tree roots produce ranked routes."""
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {"reaction_name": "Route A", "score": 0.9},
                "children": [{"smiles": "CC=O", "is_purchasable": True, "children": []}],
            },
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {"reaction_name": "Route B", "score": 0.6},
                "children": [{"smiles": "C=O", "is_purchasable": True, "children": []}],
            },
        ]
        target = TargetInput(id="ethanol", smiles="CCO")
        routes = list(adapter_instance.cast(raw_routes, target))
        assert len(routes) == 2
        assert routes[0].rank == 1
        assert routes[1].rank == 2

    def test_cycle_detection(self, adapter_instance, caplog):
        """Cyclic tree (child references parent) is caught and discarded."""
        target_smiles = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
        raw_routes = [
            {
                "smiles": target_smiles,
                "is_purchasable": False,
                "best_disconnection": {"reaction_name": "Test", "score": 0.5},
                "children": [
                    {
                        "smiles": "CC(C)c1ccccc1",
                        "is_purchasable": False,
                        "best_disconnection": {"reaction_name": "Test2", "score": 0.3},
                        "children": [
                            # Cycle: child references grandparent
                            {"smiles": target_smiles, "is_purchasable": False, "children": []},
                        ],
                    }
                ],
            }
        ]
        target = TargetInput(id="ibuprofen_cycle", smiles=canonicalize_smiles(target_smiles))
        routes = list(adapter_instance.cast(raw_routes, target))
        assert len(routes) == 0
        assert "cycle detected" in caplog.text

    def test_multi_step_tree(self, adapter_instance):
        """Two-step tree: target -> intermediate -> leaf."""
        raw_routes = [
            {
                "smiles": "CCOC(C)=O",  # ethyl acetate
                "is_purchasable": False,
                "best_disconnection": {
                    "reaction_name": "Esterification",
                    "category": "coupling",
                    "score": 0.75,
                },
                "children": [
                    {
                        "smiles": "CCO",  # ethanol (intermediate)
                        "is_purchasable": False,
                        "best_disconnection": {
                            "reaction_name": "Reduction",
                            "category": "reduction",
                            "score": 0.85,
                        },
                        "children": [
                            {"smiles": "CC=O", "is_purchasable": True, "children": []},
                        ],
                    },
                    {"smiles": "CC(O)=O", "is_purchasable": True, "children": []},  # acetic acid
                ],
            }
        ]
        target_smiles = canonicalize_smiles("CCOC(C)=O")
        target = TargetInput(id="ethyl_acetate", smiles=target_smiles)
        routes = list(adapter_instance.cast(raw_routes, target))

        assert len(routes) == 1
        route = routes[0]
        assert route.length == 2  # 2 reactions deep

        # Root step
        root_step = route.target.synthesis_step
        assert root_step is not None
        assert root_step.metadata["reaction_name"] == "Esterification"
        assert len(root_step.reactants) == 2

        # Find the intermediate (non-leaf reactant)
        intermediate = next(r for r in root_step.reactants if not r.is_leaf)
        assert intermediate.synthesis_step is not None
        assert intermediate.synthesis_step.metadata["reaction_name"] == "Reduction"

        # Leaf (starting material for inner step)
        inner_reactants = intermediate.synthesis_step.reactants
        assert len(inner_reactants) == 1
        assert inner_reactants[0].is_leaf

    def test_node_without_disconnection_is_leaf(self, adapter_instance):
        """A non-purchasable node with no children and no disconnection -> leaf."""
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "children": [],
            }
        ]
        target = TargetInput(id="ethanol", smiles="CCO")
        routes = list(adapter_instance.cast(raw_routes, target))
        # Node has no children -> leaf -> no synthesis step -> no route steps
        # The adapter treats the root as a leaf, which means Route has length 0
        assert len(routes) == 1
        assert routes[0].target.is_leaf

    def test_non_leaf_without_disconnection_raises(self, adapter_instance, caplog):
        """A non-leaf node (has children, not purchasable) without best_disconnection is an error."""
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                # No best_disconnection, but has children -> should raise
                "children": [
                    {"smiles": "CC=O", "is_purchasable": True, "children": []},
                ],
            }
        ]
        target = TargetInput(id="ethanol", smiles="CCO")
        routes = list(adapter_instance.cast(raw_routes, target))
        # Route should be discarded due to AdapterLogicError
        assert len(routes) == 0
        assert "missing 'best_disconnection'" in caplog.text

    def test_non_leaf_with_empty_reaction_name_is_discarded(self, adapter_instance, caplog):
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {
                    "reaction_name": "   ",
                    "score": 0.5,
                },
                "children": [
                    {"smiles": "CC=O", "is_purchasable": True, "children": []},
                ],
            }
        ]
        target = TargetInput(id="ethanol", smiles="CCO")

        routes = list(adapter_instance.cast(raw_routes, target))

        assert len(routes) == 0
        assert "empty 'reaction_name'" in caplog.text

    def test_invalid_top_level_payload_is_ignored(self, adapter_instance, target_input):
        raw_payload = {
            "smiles": "CCO",
            "is_purchasable": False,
            "children": [],
        }

        routes = list(adapter_instance.cast(raw_payload, target_input))

        assert routes == []

    def test_mismatched_target_smiles_route_is_discarded(self, adapter_instance):
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "best_disconnection": {
                    "reaction_name": "Reduction",
                    "score": 0.85,
                },
                "children": [
                    {"smiles": "CC=O", "is_purchasable": True, "children": []},
                ],
            }
        ]
        mismatched_target = TargetInput(id="ethanol", smiles="CCC")

        routes = list(adapter_instance.cast(raw_routes, mismatched_target))

        assert routes == []

    def test_functional_groups_in_metadata(self, adapter_instance):
        """Functional groups from MolBuilder nodes are preserved in Molecule.metadata."""
        raw_routes = [
            {
                "smiles": "CCO",
                "is_purchasable": False,
                "functional_groups": ["alcohol", "alkyl"],
                "best_disconnection": {
                    "reaction_name": "Reduction",
                    "score": 0.85,
                },
                "children": [
                    {
                        "smiles": "CC=O",
                        "is_purchasable": True,
                        "functional_groups": ["aldehyde"],
                        "children": [],
                    },
                ],
            }
        ]
        target = TargetInput(id="ethanol", smiles="CCO")
        routes = list(adapter_instance.cast(raw_routes, target))
        assert len(routes) == 1

        # Root molecule metadata
        root_mol = routes[0].target
        assert root_mol.metadata["functional_groups"] == ["alcohol", "alkyl"]

        # Leaf molecule metadata
        leaf = root_mol.synthesis_step.reactants[0]
        assert leaf.metadata["functional_groups"] == ["aldehyde"]

    def test_all_molecules_have_inchikeys(self, adapter_instance, raw_valid_route_data, target_input):
        """Every molecule in the tree should have an InChIKey."""

        def _check(mol):
            assert mol.inchikey is not None
            assert len(mol.inchikey) > 0
            if mol.synthesis_step:
                for reactant in mol.synthesis_step.reactants:
                    _check(reactant)

        routes = list(adapter_instance.cast(raw_valid_route_data, target_input))
        for route in routes:
            _check(route.target)
