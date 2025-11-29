"""Unit tests for Route.get_content_hash() method."""

import pytest

from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr

# ==============================================================================
# Route.get_content_hash Tests
# ==============================================================================


@pytest.mark.unit
class TestRouteContentHash:
    """Tests for Route.get_content_hash() method."""

    def test_get_content_hash_deterministic(self):
        """Test that get_content_hash produces deterministic results."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route = Route(target=target, rank=1)

        hash1 = route.get_content_hash()
        hash2 = route.get_content_hash()
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_get_content_hash_includes_rank(self):
        """Test that content hash differs when rank changes."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route1 = Route(target=target, rank=1)
        route2 = Route(target=target, rank=2)

        # Same tree structure, different rank = different content hash
        assert route1.get_content_hash() != route2.get_content_hash()
        # But same tree signature (structure only)
        assert route1.get_signature() == route2.get_signature()

    def test_get_content_hash_includes_metadata(self):
        """Test that content hash differs when metadata changes."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route1 = Route(target=target, rank=1, metadata={"score": 0.9})
        route2 = Route(target=target, rank=1, metadata={"score": 0.95})

        assert route1.get_content_hash() != route2.get_content_hash()

    def test_get_content_hash_includes_molecule_metadata(self):
        """Test that content hash differs when molecule metadata changes."""
        target1 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
            metadata={"price": 10.0},
        )
        target2 = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
            metadata={"price": 20.0},
        )
        route1 = Route(target=target1, rank=1)
        route2 = Route(target=target2, rank=1)

        assert route1.get_content_hash() != route2.get_content_hash()

    def test_get_content_hash_includes_reaction_details(self):
        """Test that content hash differs when reaction details change."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )

        # Route with template
        step1 = ReactionStep(reactants=[reactant], template="[C:1][O:2]>>[C:1].[O:2]")
        target1 = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
            synthesis_step=step1,
        )
        route1 = Route(target=target1, rank=1)

        # Route without template
        step2 = ReactionStep(reactants=[reactant])
        target2 = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
            synthesis_step=step2,
        )
        route2 = Route(target=target2, rank=1)

        # Different content hash because template info differs
        assert route1.get_content_hash() != route2.get_content_hash()
        # But same tree signature (structure is the same)
        assert route1.get_signature() == route2.get_signature()

    def test_get_content_hash_includes_retrocast_version(self):
        """Test that content hash includes retrocast version."""
        target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        route1 = Route(target=target, rank=1, retrocast_version="1.0.0")
        route2 = Route(target=target, rank=1, retrocast_version="2.0.0")

        assert route1.get_content_hash() != route2.get_content_hash()

    def test_get_content_hash_vs_get_signature(self):
        """Test that get_content_hash and get_signature serve different purposes."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant], template="template1", metadata={"score": 0.9})
        target = Molecule(
            smiles=SmilesStr("C"),
            inchikey=InchiKeyStr("VNWKTOKETHGBQD-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1, metadata={"cost": 100})

        signature = route.get_signature()
        content_hash = route.get_content_hash()

        # Both should be valid SHA256 hashes
        assert len(signature) == 64
        assert len(content_hash) == 64

        # But they should be different (content hash includes more info)
        assert signature != content_hash
