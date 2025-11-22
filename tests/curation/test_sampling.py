"""
Unit tests for retrocast.curation.sampling module.

Tests sampling strategies using synthetic carbon-chain routes to verify
logic without chemical complexity.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from retrocast.curation.sampling import (
    SAMPLING_STRATEGIES,
    sample_k_by_length,
    sample_random,
    sample_random_k,
    sample_stratified_priority,
    sample_top_k,
)
from retrocast.models.chem import Route

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def routes_of_varying_length(synthetic_route_factory):
    """Create routes with lengths 1, 2, 3, 4, 5."""
    return [synthetic_route_factory("linear", depth=i) for i in range(1, 6)]


@pytest.fixture
def many_routes_by_length(synthetic_route_factory):
    """
    Create multiple routes per length for testing round-robin sampling.
    3 routes each of length 1, 2, 3.
    """
    routes = []
    for depth in [1, 2, 3]:
        for i in range(3):
            route = synthetic_route_factory("linear", depth=depth)
            # Modify rank to make them distinguishable
            route = Route(target=route.target, rank=i + 1)
            routes.append(route)
    return routes


# =============================================================================
# Tests for sample_top_k
# =============================================================================


@pytest.mark.unit
class TestSampleTopK:
    def test_returns_first_k_routes(self, routes_of_varying_length):
        """Should return exactly the first k routes."""
        result = sample_top_k(routes_of_varying_length, k=3)
        assert len(result) == 3
        assert result == routes_of_varying_length[:3]

    def test_returns_all_when_k_exceeds_length(self, routes_of_varying_length):
        """Should return all routes when k > len(routes)."""
        result = sample_top_k(routes_of_varying_length, k=100)
        assert len(result) == len(routes_of_varying_length)

    def test_returns_empty_for_k_zero(self, routes_of_varying_length):
        """Should return empty list when k=0."""
        result = sample_top_k(routes_of_varying_length, k=0)
        assert result == []

    def test_returns_empty_for_negative_k(self, routes_of_varying_length):
        """Should return empty list when k<0."""
        result = sample_top_k(routes_of_varying_length, k=-1)
        assert result == []

    def test_empty_input(self):
        """Should handle empty input list."""
        result = sample_top_k([], k=5)
        assert result == []

    def test_preserves_order(self, routes_of_varying_length):
        """Should preserve original order."""
        result = sample_top_k(routes_of_varying_length, k=5)
        for i, route in enumerate(result):
            assert route.target.smiles == routes_of_varying_length[i].target.smiles


# =============================================================================
# Tests for sample_random_k
# =============================================================================


@pytest.mark.unit
class TestSampleRandomK:
    def test_returns_k_routes(self, routes_of_varying_length):
        """Should return exactly k routes."""
        result = sample_random_k(routes_of_varying_length, k=3)
        assert len(result) == 3

    def test_returns_all_when_k_exceeds_length(self, routes_of_varying_length):
        """Should return all routes when k >= len(routes)."""
        result = sample_random_k(routes_of_varying_length, k=100)
        assert len(result) == len(routes_of_varying_length)

    def test_returns_empty_for_k_zero(self, routes_of_varying_length):
        """Should return empty list when k=0."""
        result = sample_random_k(routes_of_varying_length, k=0)
        assert result == []

    def test_returns_empty_for_negative_k(self, routes_of_varying_length):
        """Should return empty list when k<0."""
        result = sample_random_k(routes_of_varying_length, k=-1)
        assert result == []

    def test_empty_input(self):
        """Should handle empty input list."""
        result = sample_random_k([], k=5)
        assert result == []

    def test_samples_are_subset(self, routes_of_varying_length):
        """All sampled routes should be from the original list."""
        result = sample_random_k(routes_of_varying_length, k=3)
        for route in result:
            assert route in routes_of_varying_length

    def test_no_duplicates(self, routes_of_varying_length):
        """Should not have duplicate routes in result."""
        result = sample_random_k(routes_of_varying_length, k=3)
        # Check by identity (same objects)
        seen = []
        for route in result:
            assert route not in seen
            seen.append(route)


# =============================================================================
# Tests for sample_k_by_length
# =============================================================================


@pytest.mark.unit
class TestSampleKByLength:
    def test_round_robin_sampling(self, many_routes_by_length):
        """Should sample one from each length group in round-robin fashion."""
        # We have 9 routes: 3 each of length 1, 2, 3
        # Sampling 3 should give us one of each length
        result = sample_k_by_length(many_routes_by_length, max_total=3)
        assert len(result) == 3

        lengths = [r.length for r in result]
        # Should have one of each length
        assert sorted(lengths) == [1, 2, 3]

    def test_prioritizes_shorter_routes(self, many_routes_by_length):
        """Should fill from shortest lengths first in each round."""
        result = sample_k_by_length(many_routes_by_length, max_total=6)
        assert len(result) == 6

        lengths = [r.length for r in result]
        # Should have 2 of each length after 2 rounds
        assert lengths.count(1) == 2
        assert lengths.count(2) == 2
        assert lengths.count(3) == 2

    def test_returns_all_when_max_exceeds_total(self, many_routes_by_length):
        """Should return all routes when max_total > len(routes)."""
        result = sample_k_by_length(many_routes_by_length, max_total=100)
        assert len(result) == len(many_routes_by_length)

    def test_returns_empty_for_zero_max(self, many_routes_by_length):
        """Should return empty list when max_total=0."""
        result = sample_k_by_length(many_routes_by_length, max_total=0)
        assert result == []

    def test_returns_empty_for_negative_max(self, many_routes_by_length):
        """Should return empty list when max_total<0."""
        result = sample_k_by_length(many_routes_by_length, max_total=-1)
        assert result == []

    def test_empty_input(self):
        """Should handle empty input list."""
        result = sample_k_by_length([], max_total=5)
        assert result == []

    def test_exhausts_short_groups_first(self, synthetic_route_factory):
        """When one group runs out, should continue with remaining groups."""
        # Create 1 route of length 1, 5 routes of length 2
        routes = [synthetic_route_factory("linear", depth=1)]
        for i in range(5):
            route = synthetic_route_factory("linear", depth=2)
            route = Route(target=route.target, rank=i + 1)
            routes.append(route)

        result = sample_k_by_length(routes, max_total=4)
        assert len(result) == 4

        lengths = [r.length for r in result]
        # Should have 1 of length 1, 3 of length 2
        assert lengths.count(1) == 1
        assert lengths.count(2) == 3


# =============================================================================
# Tests for sample_stratified_priority
# =============================================================================


@pytest.mark.unit
class TestSampleStratifiedPriority:
    def test_basic_stratified_sampling(self):
        """Should sample from pools according to counts."""
        # Create simple items with group keys
        pool1 = [1, 1, 1, 2, 2]  # Group 1: 3 items, Group 2: 2 items
        pool2 = [1, 2, 2, 2, 3, 3]  # Group 1: 1 item, Group 2: 3 items, Group 3: 2 items

        counts = {1: 2, 2: 3}  # Want 2 from group 1, 3 from group 2

        result = sample_stratified_priority(
            pools=[pool1, pool2],
            group_fn=lambda x: x,
            counts=counts,
            seed=42,
        )

        # Should have 5 items total
        assert len(result) == 5
        # Should have correct distribution
        assert result.count(1) == 2
        assert result.count(2) == 3

    def test_priority_ordering(self):
        """Should exhaust first pool before using second."""
        # Pool 1 has only 1 item in group 1
        pool1 = [1]
        # Pool 2 has more items in group 1
        pool2 = [1, 1, 1]

        counts = {1: 3}

        result = sample_stratified_priority(
            pools=[pool1, pool2],
            group_fn=lambda x: x,
            counts=counts,
            seed=42,
        )

        assert len(result) == 3

    def test_logs_warning_when_insufficient_items(self, caplog):
        """Should log warning when not enough items across all pools."""
        pool1 = [1]
        pool2 = [1]

        counts = {1: 5}  # Want 5 but only have 2

        with caplog.at_level("WARNING"):
            result = sample_stratified_priority(
                pools=[pool1, pool2],
                group_fn=lambda x: x,
                counts=counts,
                seed=42,
            )

        # Should return what it could find (2 items)
        assert len(result) == 2
        # Should have logged a warning
        assert "Cannot sample 5 items for group 1" in caplog.text
        assert "only found 2 across all pools" in caplog.text

    def test_deterministic_with_seed(self):
        """Should produce same results with same seed."""
        pool = [1, 1, 1, 1, 1]
        counts = {1: 3}

        result1 = sample_stratified_priority(
            pools=[pool],
            group_fn=lambda x: x,
            counts=counts,
            seed=42,
        )
        result2 = sample_stratified_priority(
            pools=[pool],
            group_fn=lambda x: x,
            counts=counts,
            seed=42,
        )

        assert result1 == result2

    def test_ignores_groups_not_in_counts(self):
        """Should ignore items whose group is not in counts."""
        pool = [1, 1, 2, 2, 3, 3]
        counts = {1: 2}  # Only want group 1

        result = sample_stratified_priority(
            pools=[pool],
            group_fn=lambda x: x,
            counts=counts,
            seed=42,
        )

        assert len(result) == 2
        assert all(x == 1 for x in result)

    def test_with_route_objects(self, many_routes_by_length):
        """Should work with actual Route objects using length as group key."""
        counts = {1: 1, 2: 1, 3: 1}

        result = sample_stratified_priority(
            pools=[many_routes_by_length],
            group_fn=lambda r: r.length,
            counts=counts,
            seed=42,
        )

        assert len(result) == 3
        lengths = [r.length for r in result]
        assert sorted(lengths) == [1, 2, 3]


# =============================================================================
# Tests for sample_random
# =============================================================================


@pytest.mark.unit
class TestSampleRandom:
    def test_returns_n_items(self):
        """Should return exactly n items."""
        items = [1, 2, 3, 4, 5]
        result = sample_random(items, n=3, seed=42)
        assert len(result) == 3

    def test_raises_when_n_exceeds_length(self):
        """Should raise ValueError when n > len(items)."""
        items = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot sample 5 from 3 items"):
            sample_random(items, n=5, seed=42)

    def test_deterministic_with_seed(self):
        """Should produce same results with same seed."""
        items = [1, 2, 3, 4, 5]
        result1 = sample_random(items, n=3, seed=42)
        result2 = sample_random(items, n=3, seed=42)
        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds should (usually) produce different results."""
        items = list(range(100))
        result1 = sample_random(items, n=10, seed=1)
        result2 = sample_random(items, n=10, seed=2)
        assert result1 != result2

    def test_samples_are_subset(self):
        """All sampled items should be from the original list."""
        items = [1, 2, 3, 4, 5]
        result = sample_random(items, n=3, seed=42)
        for item in result:
            assert item in items


# =============================================================================
# Tests for SAMPLING_STRATEGIES registry
# =============================================================================


@pytest.mark.unit
class TestSamplingStrategies:
    def test_strategies_registered(self):
        """All expected strategies should be registered."""
        assert "top-k" in SAMPLING_STRATEGIES
        assert "random-k" in SAMPLING_STRATEGIES
        assert "by-length" in SAMPLING_STRATEGIES

    def test_strategy_callables(self, routes_of_varying_length):
        """Registered strategies should be callable."""
        for _name, fn in SAMPLING_STRATEGIES.items():
            result = fn(routes_of_varying_length, 2)
            assert isinstance(result, list)


# =============================================================================
# Hypothesis property-based tests
# =============================================================================


@pytest.mark.unit
class TestSamplingProperties:
    @given(
        k=st.integers(min_value=0, max_value=20),
        n_routes=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_top_k_length_invariant(self, k, n_routes, synthetic_route_factory):
        """sample_top_k should return min(k, n_routes) items."""
        if n_routes == 0:
            routes = []
        else:
            routes = [synthetic_route_factory("linear", depth=1) for _ in range(n_routes)]

        result = sample_top_k(routes, k)
        expected_len = min(k, n_routes) if k > 0 else 0
        assert len(result) == expected_len

    @given(
        k=st.integers(min_value=0, max_value=20),
        n_routes=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_k_length_invariant(self, k, n_routes, synthetic_route_factory):
        """sample_random_k should return min(k, n_routes) items."""
        routes = [synthetic_route_factory("linear", depth=1) for _ in range(n_routes)]

        result = sample_random_k(routes, k)
        expected_len = min(k, n_routes) if k > 0 else 0
        assert len(result) == expected_len
