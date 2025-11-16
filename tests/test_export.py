"""Tests for retrocast.export module - reaction SMILES extraction and export."""

import gzip

from retrocast.export import (
    collect_reaction_smiles_and_signatures,
    extract_reactions_from_routes,
)
from retrocast.io import save_reaction_smiles
from retrocast.schemas import Molecule, ReactionSignature, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr


class TestExtractReactionsFromRoutes:
    """Tests for extracting reactions from multiple routes with deduplication."""

    def test_empty_routes_dict(self):
        """Empty routes dict should return empty set."""
        routes: dict[str, list[Route]] = {}
        reactions = extract_reactions_from_routes(routes)
        assert reactions == set()

    def test_single_route_single_target(self):
        """Single route should extract its reactions."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant])
        target = Molecule(
            smiles=SmilesStr("CC=O"),
            inchikey=InchiKeyStr("IKHGUXGNUITLKF-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)

        routes = {"target1": [route]}
        reactions = extract_reactions_from_routes(routes)

        assert len(reactions) == 1
        assert "CCO>>CC=O" in reactions

    def test_multiple_routes_deduplication(self):
        """Same reaction in different routes should be deduplicated."""
        # Two routes that share the same reaction
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )

        target1 = Molecule(
            smiles=SmilesStr("CC=O"),
            inchikey=InchiKeyStr("KEY-1"),
            synthesis_step=ReactionStep(reactants=[reactant]),
        )
        target2 = Molecule(
            smiles=SmilesStr("CC=O"),
            inchikey=InchiKeyStr("KEY-1"),  # Same reaction
            synthesis_step=ReactionStep(reactants=[reactant]),
        )

        routes = {
            "target1": [Route(target=target1, rank=1)],
            "target2": [Route(target=target2, rank=1)],
        }
        reactions = extract_reactions_from_routes(routes)

        # Same reaction should be deduplicated
        assert len(reactions) == 1
        assert "CCO>>CC=O" in reactions

    def test_exclude_reactions(self):
        """Reactions in exclusion set should be excluded."""
        # Route with two reactions: A -> B -> C
        leaf_a = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-A"))
        intermediate_b = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("KEY-B"),
            synthesis_step=ReactionStep(reactants=[leaf_a]),
        )
        target_c = Molecule(
            smiles=SmilesStr("COC"),
            inchikey=InchiKeyStr("KEY-C"),
            synthesis_step=ReactionStep(reactants=[intermediate_b]),
        )
        route = Route(target=target_c, rank=1)

        # Exclude the first reaction: A -> B
        exclude: set[ReactionSignature] = {(frozenset(["KEY-A"]), "KEY-B")}

        routes = {"target1": [route]}
        reactions = extract_reactions_from_routes(routes, exclude_reactions=exclude)

        # Only B -> C should remain
        assert len(reactions) == 1
        assert "CO>>COC" in reactions
        assert "C>>CO" not in reactions

    def test_multiple_targets_multiple_routes(self):
        """Multiple targets with multiple routes each."""
        # Target 1: two routes
        r1_reactant = Molecule(smiles=SmilesStr("C"), inchikey=InchiKeyStr("KEY-1"))
        r1_target = Molecule(
            smiles=SmilesStr("CO"),
            inchikey=InchiKeyStr("KEY-2"),
            synthesis_step=ReactionStep(reactants=[r1_reactant]),
        )
        route1 = Route(target=r1_target, rank=1)

        r2_reactant = Molecule(smiles=SmilesStr("CC"), inchikey=InchiKeyStr("KEY-3"))
        r2_target = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("KEY-4"),
            synthesis_step=ReactionStep(reactants=[r2_reactant]),
        )
        route2 = Route(target=r2_target, rank=2)

        # Target 2: one route
        r3_reactant = Molecule(smiles=SmilesStr("CCC"), inchikey=InchiKeyStr("KEY-5"))
        r3_target = Molecule(
            smiles=SmilesStr("CCCO"),
            inchikey=InchiKeyStr("KEY-6"),
            synthesis_step=ReactionStep(reactants=[r3_reactant]),
        )
        route3 = Route(target=r3_target, rank=1)

        routes = {
            "target1": [route1, route2],
            "target2": [route3],
        }
        reactions = extract_reactions_from_routes(routes)

        assert len(reactions) == 3
        assert "C>>CO" in reactions
        assert "CC>>CCO" in reactions
        assert "CCC>>CCCO" in reactions


class TestExtractReactionsWithSignatures:
    """Tests for extracting reactions with their signatures."""

    def test_single_step_returns_smiles_and_signature(self):
        """Should return both SMILES and signature in one pass."""
        reactant = Molecule(
            smiles=SmilesStr("CCO"),
            inchikey=InchiKeyStr("LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        )
        step = ReactionStep(reactants=[reactant])
        target = Molecule(
            smiles=SmilesStr("CC=O"),
            inchikey=InchiKeyStr("IKHGUXGNUITLKF-UHFFFAOYSA-N"),
            synthesis_step=step,
        )
        route = Route(target=target, rank=1)

        result = collect_reaction_smiles_and_signatures(route)

        assert len(result) == 1
        rxn_smiles, sig = result[0]
        assert rxn_smiles == "CCO>>CC=O"
        assert sig == (frozenset(["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]), "IKHGUXGNUITLKF-UHFFFAOYSA-N")


class TestSaveReactionSmiles:
    """Tests for saving reaction SMILES to file (I/O layer)."""

    def test_save_creates_file(self, tmp_path):
        """Save should create a gzipped text file."""
        reactions = {"CCO>>CC=O"}
        output_path = tmp_path / "reactions.txt.gz"
        save_reaction_smiles(reactions, output_path)

        assert output_path.exists()
        assert output_path.suffix == ".gz"

    def test_save_file_content_format(self, tmp_path):
        """Each reaction should be on its own line."""
        reactions = {"C>>CO", "CO>>COC"}
        output_path = tmp_path / "reactions.txt.gz"
        save_reaction_smiles(reactions, output_path)

        with gzip.open(output_path, "rt") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 2
        assert set(lines) == reactions

    def test_save_empty_set(self, tmp_path):
        """Empty set should create file with no content."""
        reactions: set[str] = set()
        output_path = tmp_path / "reactions.txt.gz"
        save_reaction_smiles(reactions, output_path)

        with gzip.open(output_path, "rt") as f:
            content = f.read().strip()

        assert content == ""

    def test_save_sorted_output(self, tmp_path):
        """Output should be sorted for reproducibility."""
        reactions = {"CCC>>CCCO", "C>>CO", "CC>>CCO"}
        output_path = tmp_path / "reactions.txt.gz"
        save_reaction_smiles(reactions, output_path)

        with gzip.open(output_path, "rt") as f:
            lines = f.read().strip().split("\n")

        assert lines == sorted(lines)

    def test_save_accepts_list(self, tmp_path):
        """Should accept a list as well as a set."""
        reactions = ["C>>CO", "CC>>CCO"]
        output_path = tmp_path / "reactions.txt.gz"
        save_reaction_smiles(reactions, output_path)

        with gzip.open(output_path, "rt") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 2
