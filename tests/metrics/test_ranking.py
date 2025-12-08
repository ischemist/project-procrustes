"""
Unit tests for retrocast.metrics.ranking module.

Tests probabilistic ranking and pairwise tournament using synthetic evaluation data.
"""

import pytest

from retrocast.metrics.ranking import compute_pairwise_tournament, compute_probabilistic_ranking
from retrocast.models.evaluation import EvaluationResults, ScoredRoute, TargetEvaluation
from retrocast.models.stats import ModelComparison

# =============================================================================
# Fixtures
# =============================================================================


def _make_target_evaluation(
    target_id: str,
    solvable: bool,
    acceptable_rank: int | None = None,
    route_length: int = 1,
    is_convergent: bool = False,
) -> TargetEvaluation:
    """Create a synthetic TargetEvaluation with optional solved route."""
    routes = []
    if solvable:
        # Add a solved route at acceptable_rank position
        if acceptable_rank is None:
            acceptable_rank = 1
        # Add solved route at position acceptable_rank
        for i in range(1, acceptable_rank + 1):
            is_solved = i == acceptable_rank
            routes.append(
                ScoredRoute(
                    rank=i,
                    is_solved=is_solved,
                    matches_acceptable=is_solved,
                )
            )
        # Add more unsolved routes for realism
        for i in range(acceptable_rank + 1, acceptable_rank + 4):
            routes.append(
                ScoredRoute(
                    rank=i,
                    is_solved=False,
                    matches_acceptable=False,
                )
            )

    return TargetEvaluation(
        target_id=target_id,
        routes=routes,
        is_solvable=solvable,
        acceptable_rank=acceptable_rank,
        stratification_length=route_length,
        stratification_is_convergent=is_convergent,
    )


@pytest.fixture
def three_models_perfect_data():
    """
    Create three models with synthetic evaluation results where:
    - Model A: perfect (all targets solvable at rank 1)
    - Model B: good (90% solvable at rank 1)
    - Model C: bad (50% solvable at rank 1)
    """
    n_targets = 10

    # Model A: all solved at rank 1
    results_a = {}
    for i in range(n_targets):
        results_a[f"target_{i}"] = _make_target_evaluation(
            f"target_{i}",
            solvable=True,
            acceptable_rank=1,
        )

    # Model B: 9/10 solved at rank 1, 1/10 at rank 2
    results_b = {}
    for i in range(n_targets):
        acceptable_rank = 1 if i < 9 else 2
        results_b[f"target_{i}"] = _make_target_evaluation(
            f"target_{i}",
            solvable=True,
            acceptable_rank=acceptable_rank,
        )

    # Model C: 5/10 solved at rank 1, others unsolved
    results_c = {}
    for i in range(n_targets):
        solvable = i < 5
        acceptable_rank = 1 if solvable else None
        results_c[f"target_{i}"] = _make_target_evaluation(
            f"target_{i}",
            solvable=solvable,
            acceptable_rank=acceptable_rank,
        )

        return {
            "model_a": EvaluationResults(
                model_name="model_a",
                benchmark_name="test_bench",
                stock_name="test_stock",
                has_acceptable_routes=True,
                results=results_a,
            ),
            "model_b": EvaluationResults(
                model_name="model_b",
                benchmark_name="test_bench",
                stock_name="test_stock",
                has_acceptable_routes=True,
                results=results_b,
            ),
            "model_c": EvaluationResults(
                model_name="model_c",
                benchmark_name="test_bench",
                stock_name="test_stock",
                has_acceptable_routes=True,
                results=results_c,
            ),
        }


@pytest.fixture
def two_models_identical():
    """Create two models with identical results."""
    n_targets = 5

    results_shared = {}
    for i in range(n_targets):
        results_shared[f"target_{i}"] = _make_target_evaluation(
            f"target_{i}",
            solvable=True,
            acceptable_rank=1,
        )

    return {
        "model_x": EvaluationResults(
            model_name="model_x",
            benchmark_name="test",
            stock_name="test",
            has_acceptable_routes=True,
            results=results_shared.copy(),
        ),
        "model_y": EvaluationResults(
            model_name="model_y",
            benchmark_name="test",
            stock_name="test",
            has_acceptable_routes=True,
            results=results_shared.copy(),
        ),
    }


# =============================================================================
# Tests for compute_probabilistic_ranking
# =============================================================================


@pytest.mark.unit
class TestComputeProbabilisticRanking:
    def test_single_model(self):
        """Single model should have 100% probability of rank 1."""
        results = {
            "model_only": EvaluationResults(
                model_name="model_only",
                benchmark_name="test",
                stock_name="test",
                has_acceptable_routes=True,
                results={
                    "t1": _make_target_evaluation("t1", solvable=True, acceptable_rank=1),
                    "t2": _make_target_evaluation("t2", solvable=True, acceptable_rank=1),
                },
            )
        }

        # Simple metric: is_solvable as 0/1
        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(results, solvability, n_boot=100, seed=42)

        assert len(ranking) == 1
        assert ranking[0].model_name == "model_only"
        assert ranking[0].expected_rank == 1.0
        assert ranking[0].rank_probs[1] == 1.0

    def test_three_models_ranked(self, three_models_perfect_data):
        """Three models should be ranked by solvability rate."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=1000, seed=42)

        # Should have 3 results
        assert len(ranking) == 3

        # Should be sorted by expected rank
        expected_order = ["model_a", "model_b", "model_c"]
        actual_order = [r.model_name for r in ranking]
        assert actual_order == expected_order

        # Model A should have better expected rank than B and C
        assert ranking[0].expected_rank < ranking[1].expected_rank
        assert ranking[1].expected_rank < ranking[2].expected_rank

    def test_rank_probabilities_sum_to_one(self, three_models_perfect_data):
        """For each model, rank probabilities should sum to ~1.0."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)

        for result in ranking:
            total_prob = sum(result.rank_probs.values())
            assert abs(total_prob - 1.0) < 0.01  # Allow small rounding error

    def test_rank_probabilities_valid_range(self, three_models_perfect_data):
        """All rank probabilities should be in [0, 1]."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)

        for result in ranking:
            for rank, prob in result.rank_probs.items():
                assert 0.0 <= prob <= 1.0
                assert 1 <= rank <= len(three_models_perfect_data)

    def test_expected_rank_consistency(self, three_models_perfect_data):
        """Expected rank should be mean of rank probabilities."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)

        for result in ranking:
            calculated_expected = sum(rank * prob for rank, prob in result.rank_probs.items())
            assert abs(result.expected_rank - calculated_expected) < 0.01

    def test_deterministic_with_seed(self, three_models_perfect_data):
        """Results should be deterministic with same seed."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking1 = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)
        ranking2 = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)

        for r1, r2 in zip(ranking1, ranking2, strict=True):
            assert r1.model_name == r2.model_name
            assert r1.expected_rank == r2.expected_rank
            assert r1.rank_probs == r2.rank_probs

    def test_different_seeds_different_results(self, three_models_perfect_data):
        """Different seeds may produce different results (but with clear winners, may be stable)."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking1 = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=1)
        ranking2 = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=2)

        # Ordering should be same (clear winners)
        order1 = [r.model_name for r in ranking1]
        order2 = [r.model_name for r in ranking2]
        assert order1 == order2

    def test_gt_rank_metric(self, three_models_perfect_data):
        """Test with acceptable rank as metric."""

        def acceptable_rank_metric(te: TargetEvaluation) -> float:
            # Convert to metric where lower is better (like rank position)
            if te.acceptable_rank is None:
                return 0.0  # Worst possible (no solution)
            return 1.0 / te.acceptable_rank  # Reciprocal so higher is better for boot dist

        ranking = compute_probabilistic_ranking(three_models_perfect_data, acceptable_rank_metric, n_boot=100, seed=42)

        assert len(ranking) == 3
        # Model A should be in top rankings (all at rank 1 = best)
        names = [r.model_name for r in ranking]
        assert "model_a" in names


# =============================================================================
# Tests for compute_pairwise_tournament
# =============================================================================


@pytest.mark.unit
class TestComputePairwiseTournament:
    def test_two_models(self, two_models_identical):
        """Two models with identical results should have ~0 difference."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(two_models_identical, solvability, "solvability", n_boot=100)

        # Should have 2 comparisons: X vs Y and Y vs X
        assert len(comparisons) == 2

        # Both should show ~0 difference
        for comp in comparisons:
            assert abs(comp.diff_mean) < 0.1  # Small difference

    def test_comparison_count(self, three_models_perfect_data):
        """With N models, should have N*(N-1) comparisons (directed)."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=50)

        n_models = len(three_models_perfect_data)
        expected_comparisons = n_models * (n_models - 1)
        assert len(comparisons) == expected_comparisons

    def test_comparison_symmetry(self, three_models_perfect_data):
        """If A vs B has diff=d, B vs A should have diff=-d."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=100)

        # Group by model pairs
        comp_dict = {}
        for comp in comparisons:
            key = (comp.model_a, comp.model_b)
            comp_dict[key] = comp

        # Check a few pairs
        for (a, b), comp_ab in comp_dict.items():
            if (b, a) in comp_dict:
                comp_ba = comp_dict[(b, a)]
                # B vs A should be roughly negative of A vs B
                assert abs(comp_ab.diff_mean + comp_ba.diff_mean) < 0.2

    def test_comparison_has_required_fields(self, three_models_perfect_data):
        """Each comparison should have all required fields."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=50)

        assert len(comparisons) > 0

        for comp in comparisons:
            assert isinstance(comp, ModelComparison)
            assert comp.metric == "solvability"
            assert comp.model_a is not None
            assert comp.model_b is not None
            assert isinstance(comp.diff_mean, float)
            assert isinstance(comp.diff_ci_lower, float)
            assert isinstance(comp.diff_ci_upper, float)
            assert isinstance(comp.is_significant, bool)

    def test_ci_bounds_consistent(self, three_models_perfect_data):
        """CI lower should be <= mean <= CI upper."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=100)

        for comp in comparisons:
            assert comp.diff_ci_lower <= comp.diff_mean <= comp.diff_ci_upper

    def test_tournament_result_types(self, three_models_perfect_data):
        """Should be able to rank models based on tournament results."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=50)

        assert all(isinstance(c, ModelComparison) for c in comparisons)

    def test_identical_models_not_significant(self, two_models_identical):
        """Comparisons of identical models should not be significant."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        comparisons = compute_pairwise_tournament(two_models_identical, solvability, "solvability", n_boot=200)

        # Most comparisons should not be significant (or marginally so)
        non_significant = [c for c in comparisons if not c.is_significant]
        assert len(non_significant) >= len(comparisons) // 2


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.integration
class TestRankingIntegration:
    def test_full_ranking_workflow(self, three_models_perfect_data):
        """Test full workflow: ranking then pairwise comparison."""

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        # Step 1: Get probabilistic ranking
        ranking = compute_probabilistic_ranking(three_models_perfect_data, solvability, n_boot=100, seed=42)

        assert len(ranking) == 3

        # Step 2: Run pairwise tournament
        comparisons = compute_pairwise_tournament(three_models_perfect_data, solvability, "solvability", n_boot=100)

        assert len(comparisons) == 6  # 3 * (3-1)

        # Step 3: Verify consistency - best ranked model has best expected rank
        best_expected_rank = ranking[0].expected_rank
        # Best model should have lower expected rank than others
        other_ranks = [r.expected_rank for r in ranking[1:]]
        assert all(best_expected_rank <= r for r in other_ranks)

    def test_ranking_with_partial_solvability(self):
        """Test ranking with varying solvability across models."""
        # Model A: 80% solvable
        results_a = {f"t{i}": _make_target_evaluation(f"t{i}", solvable=i < 8) for i in range(10)}

        # Model B: 60% solvable
        results_b = {f"t{i}": _make_target_evaluation(f"t{i}", solvable=i < 6) for i in range(10)}

        models = {
            "a": EvaluationResults(
                model_name="a",
                benchmark_name="test",
                stock_name="test",
                has_acceptable_routes=True,
                results=results_a,
            ),
            "b": EvaluationResults(
                model_name="b",
                benchmark_name="test",
                stock_name="test",
                has_acceptable_routes=True,
                results=results_b,
            ),
        }

        def solvability(te: TargetEvaluation) -> float:
            return 1.0 if te.is_solvable else 0.0

        ranking = compute_probabilistic_ranking(models, solvability, n_boot=100, seed=42)

        # A should rank higher than B
        assert ranking[0].model_name == "a"
        assert ranking[0].expected_rank < ranking[1].expected_rank
