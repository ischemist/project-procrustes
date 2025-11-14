import pytest

from retrocast.domain.schemas import BenchmarkTree, MoleculeNode, ReactionNode, RunStatistics, TargetInfo


@pytest.fixture
def populated_stats() -> RunStatistics:
    """Provides a RunStatistics object with typical, non-trivial data."""
    return RunStatistics(
        total_routes_in_raw_files=100,
        routes_failed_validation=10,
        routes_failed_transformation=5,
        successful_routes_before_dedup=85,
        final_unique_routes_saved=60,
        targets_with_at_least_one_route={"target_a", "target_b", "target_c"},
        routes_per_target={"target_a": 10, "target_b": 20, "target_c": 30},
    )


class TestRunStatistics:
    def test_properties_on_populated_stats(self, populated_stats: RunStatistics):
        """Tests that all calculated properties are correct for a typical run."""
        assert populated_stats.total_failures == 15
        assert populated_stats.num_targets_with_routes == 3
        assert populated_stats.duplication_factor == round(85 / 60, 2)
        assert populated_stats.min_routes_per_target == 10
        assert populated_stats.max_routes_per_target == 30
        assert populated_stats.avg_routes_per_target == 20.0
        assert populated_stats.median_routes_per_target == 20.0

    def test_properties_on_empty_stats(self):
        """Tests that properties return sane defaults for an empty stats object."""
        stats = RunStatistics()
        assert stats.total_failures == 0
        assert stats.num_targets_with_routes == 0
        assert stats.duplication_factor == 0.0
        assert stats.min_routes_per_target == 0
        assert stats.max_routes_per_target == 0
        assert stats.avg_routes_per_target == 0.0
        assert stats.median_routes_per_target == 0.0

    def test_duplication_factor_handles_zero_division(self):
        """Ensures duplication_factor returns 0.0 if no routes were saved."""
        stats = RunStatistics(successful_routes_before_dedup=10, final_unique_routes_saved=0)
        assert stats.duplication_factor == 0.0

    def test_to_manifest_dict(self, populated_stats: RunStatistics):
        """Verifies the structure and values of the manifest dictionary."""
        manifest = populated_stats.to_manifest_dict()
        assert manifest["final_unique_routes_saved"] == 60
        assert manifest["num_targets_with_at_least_one_route"] == 3
        assert manifest["avg_routes_per_target"] == 20.0
        # Check the combined failure/duplicate count
        num_duplicates = (
            populated_stats.successful_routes_before_dedup - populated_stats.final_unique_routes_saved
        )  # 85 - 60 = 25
        total_failures = populated_stats.total_failures  # 15
        assert manifest["total_routes_failed_or_duplicate"] == num_duplicates + total_failures


@pytest.fixture(scope="module")
def multi_level_tree() -> BenchmarkTree:
    """
    Builds a consistent, multi-level tree for testing traversal methods.
    Structure: T -> (I1, S1), where I1 -> (S2, S3).
    S1, S2, S3 are starting materials. Longest path is 2 steps.
    """
    # Level 2 (Starting Materials)
    s2 = MoleculeNode(id="s2", molecule_hash="h_s2", smiles="S2", is_starting_material=True)
    s3 = MoleculeNode(id="s3", molecule_hash="h_s3", smiles="S3", is_starting_material=True)

    # Level 1 (Intermediate)
    rxn1 = ReactionNode(id="rxn1", reaction_smiles="S2.S3>>I1", reactants=[s2, s3])
    i1 = MoleculeNode(id="i1", molecule_hash="h_i1", smiles="I1", is_starting_material=False, reactions=[rxn1])

    # Another starting material at level 1
    s1 = MoleculeNode(id="s1", molecule_hash="h_s1", smiles="S1", is_starting_material=True)

    # Level 0 (Target)
    # sorting alphabetically: I1, S1
    reactants_sorted = sorted([i1, s1], key=lambda n: n.smiles)
    rxn_root = ReactionNode(id="rxn_root", reaction_smiles="I1.S1>>T", reactants=reactants_sorted)
    root = MoleculeNode(id="root", molecule_hash="h_t", smiles="T", is_starting_material=False, reactions=[rxn_root])

    target_info = TargetInfo(id="target_t", smiles="T")
    return BenchmarkTree(target=target_info, retrosynthetic_tree=root)


class TestTreeSchemas:
    def test_get_depth(self, multi_level_tree: BenchmarkTree):
        """Tests depth calculation on various nodes within a tree."""
        root = multi_level_tree.retrosynthetic_tree
        i1 = root.reactions[0].reactants[0]  # Intermediate I1
        s1 = root.reactions[0].reactants[1]  # Starting material S1
        s2 = i1.reactions[0].reactants[0]  # Starting material S2

        assert root.get_depth() == 2
        assert i1.get_depth() == 1
        assert s1.get_depth() == 0
        assert s2.get_depth() == 0

    def test_molecule_node_to_simple_tree(self, multi_level_tree: BenchmarkTree):
        """Tests the conversion of a MoleculeNode to a simple nested dict."""
        root = multi_level_tree.retrosynthetic_tree
        simple_tree = root.to_simple_tree()

        expected_structure = {
            "smiles": "T",
            "children": [
                {
                    "smiles": "I1",
                    "children": [
                        {"smiles": "S2", "children": []},
                        {"smiles": "S3", "children": []},
                    ],
                },
                {"smiles": "S1", "children": []},
            ],
        }
        assert simple_tree == expected_structure

    def test_benchmark_tree_to_simple_tree(self, multi_level_tree: BenchmarkTree):
        """Ensures the BenchmarkTree's delegation method works correctly."""
        # This just calls the root node's method, so it should be identical.
        simple_tree = multi_level_tree.to_simple_tree()
        expected_structure = multi_level_tree.retrosynthetic_tree.to_simple_tree()
        assert simple_tree == expected_structure
