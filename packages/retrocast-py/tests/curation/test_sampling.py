from __future__ import annotations

import pytest

from retrocast.curation.sampling import sample_random, sample_stratified_priority


def test_sample_stratified_priority_exhausts_pools_in_order() -> None:
    primary = ["p0-a", "p0-b", "p1-a"]
    secondary = ["s0-a", "s0-b", "s1-a"]
    groups = {
        "p0-a": 0,
        "p0-b": 0,
        "p1-a": 1,
        "s0-a": 0,
        "s0-b": 0,
        "s1-a": 1,
    }

    selected = sample_stratified_priority(
        [primary, secondary],
        lambda value: groups[value],
        {0: 3, 1: 2},
        seed=1,
    )

    assert selected == ["p0-a", "p0-b", "s0-a", "p1-a", "s1-a"]


def test_sample_stratified_priority_samples_within_group_deterministically() -> None:
    values = list(range(10))

    selected = sample_stratified_priority([values], lambda value: value % 2, {0: 3, 1: 2}, seed=7)

    assert selected == [4, 2, 6, 1, 9]


def test_sample_stratified_priority_takes_all_when_pool_is_short() -> None:
    selected = sample_stratified_priority([[1], [2]], lambda value: 0, {0: 5}, seed=1)

    assert selected == [1, 2]


def test_sample_random_is_seeded_and_rejects_oversampling() -> None:
    assert sample_random([1, 2, 3, 4], 2, seed=1) == [2, 3]

    with pytest.raises(ValueError, match="cannot sample 5 from 4 items"):
        sample_random([1, 2, 3, 4], 5, seed=1)
