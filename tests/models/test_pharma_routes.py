"""Contract/Regression tests using real pharmaceutical route data."""

from typing import Any

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.models.chem import Molecule, ReactionStep, Route

# ==============================================================================
# Contract/Regression Tests using Pharma Routes
# ==============================================================================


@pytest.mark.contract
@pytest.mark.regression
class TestPharmaRoutesContract:
    """Contract tests using real pharmaceutical route data."""

    def _build_molecule_tree(self, node_data: dict[str, Any]) -> Molecule:
        """Recursively build a Molecule tree from pharma routes JSON structure."""
        smiles = canonicalize_smiles(node_data["smiles"])
        inchikey = get_inchi_key(smiles)

        # Check if this node has children (i.e., it's not a leaf)
        if "children" in node_data and node_data["children"]:
            # Recursively build reactant molecules
            reactants = [self._build_molecule_tree(child) for child in node_data["children"]]
            synthesis_step = ReactionStep(reactants=reactants)
            return Molecule(smiles=smiles, inchikey=inchikey, synthesis_step=synthesis_step)
        else:
            # Leaf node
            return Molecule(smiles=smiles, inchikey=inchikey)

    def test_vonoprazan_route_structure(self, pharma_routes_data: dict[str, Any]):
        """Test building and analyzing vonoprazan-1 route."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        target_molecule = self._build_molecule_tree(vonoprazan_data)
        route = Route(target=target_molecule, rank=1)

        # Verify route properties
        assert route.rank == 1
        assert not route.target.is_leaf
        assert route.length > 0

        # Expected leaves based on pharma_routes.json structure
        # Vonoprazan has 3 leaf nodes: "O=Cc1c[nH]c(-c2ccccc2F)c1", "O=S(=O)(Cl)c1cccnc1", "CN"
        assert len(route.leaves) == 3

        # Verify signature is deterministic
        sig1 = route.get_signature()
        sig2 = route.get_signature()
        assert sig1 == sig2

    def test_mitapivat_route_structure(self, pharma_routes_data: dict[str, Any]):
        """Test building and analyzing mitapivat-1 route (deeper tree)."""
        mitapivat_data = pharma_routes_data["mitapivat-1"]
        target_molecule = self._build_molecule_tree(mitapivat_data)
        route = Route(target=target_molecule, rank=1)

        # Verify route properties
        assert route.rank == 1
        assert not route.target.is_leaf
        assert route.length >= 3  # Mitapivat has a deeper tree

        # Count leaves
        leaves = route.leaves
        assert len(leaves) > 0

        # All leaves should be leaf molecules
        for leaf in leaves:
            assert leaf.is_leaf

    def test_pharma_routes_signature_uniqueness(self, pharma_routes_data: dict[str, Any]):
        """Test that different pharma routes have different signatures."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        mitapivat_data = pharma_routes_data["mitapivat-1"]

        vonoprazan_target = self._build_molecule_tree(vonoprazan_data)
        mitapivat_target = self._build_molecule_tree(mitapivat_data)

        route1 = Route(target=vonoprazan_target, rank=1)
        route2 = Route(target=mitapivat_target, rank=1)

        # Different routes should have different signatures
        assert route1.get_signature() != route2.get_signature()

    def test_pharma_routes_roundtrip(self, pharma_routes_data: dict[str, Any]):
        """Test that pharma routes can be built, serialized, and reconstructed."""
        for route_id, route_data in pharma_routes_data.items():
            # Build route from JSON data
            target_molecule = self._build_molecule_tree(route_data)
            original_route = Route(target=target_molecule, rank=1)

            # Serialize to dict
            route_dict = original_route.model_dump(exclude={"leaves", "depth"})
            assert "target" in route_dict
            assert "rank" in route_dict
            assert route_dict["rank"] == 1

            # Reconstruct from dict (this is the "round trip")
            reconstructed_route = Route.model_validate(route_dict)

            # Verify reconstructed route matches original
            assert reconstructed_route.rank == original_route.rank
            assert reconstructed_route.target.smiles == original_route.target.smiles
            assert reconstructed_route.target.inchikey == original_route.target.inchikey

            # Verify signatures match (proves tree structure is preserved)
            assert reconstructed_route.get_signature() == original_route.get_signature(), (
                f"Route {route_id}: signatures don't match after roundtrip"
            )

            # Verify target SMILES matches original data (after canonicalization)
            expected_smiles = canonicalize_smiles(route_data["smiles"])
            assert reconstructed_route.target.smiles == expected_smiles

    def test_vonoprazan_depth_calculation(self, pharma_routes_data: dict[str, Any]):
        """Test specific depth calculation for vonoprazan route."""
        vonoprazan_data = pharma_routes_data["vonoprazan-1"]
        target_molecule = self._build_molecule_tree(vonoprazan_data)
        route = Route(target=target_molecule, rank=1)

        # Based on the structure in pharma_routes.json:
        # vonoprazan-1 has 2 levels of reactions
        # Level 1: target -> [intermediate, "CN"]
        # Level 2: intermediate -> ["O=Cc1c[nH]c(-c2ccccc2F)c1", "O=S(=O)(Cl)c1cccnc1"]
        assert route.length == 2

    def test_mitapivat_depth_calculation(self, pharma_routes_data: dict[str, Any]):
        """Test specific depth calculation for mitapivat route."""
        mitapivat_data = pharma_routes_data["mitapivat-1"]
        target_molecule = self._build_molecule_tree(mitapivat_data)
        route = Route(target=target_molecule, rank=1)

        # Mitapivat has a deeper tree structure
        # Should have depth >= 4 based on the nested structure
        assert route.length >= 4
