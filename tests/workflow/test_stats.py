from retrocast.models.candidates import Candidate, FailureRecord
from retrocast.typing import ErrorCode
from retrocast.workflow.stats import candidate_statistics, collected_candidate_statistics


def failed_candidate(rank: int, code: str) -> Candidate:
    return Candidate(rank=rank, failure=FailureRecord(code=ErrorCode(code), target_id="target"))


def test_candidate_statistics_are_computed_by_the_native_engine() -> None:
    statistics = candidate_statistics(
        [
            failed_candidate(1, "adapter.schema_invalid"),
            failed_candidate(2, "adapter.schema_invalid"),
            failed_candidate(3, "adapter.route_invalid"),
        ]
    )

    assert statistics.to_manifest_dict() == {
        "total_candidates_seen": 3,
        "successful_candidates": 0,
        "failed_candidates": 3,
        "final_candidates_saved": 3,
        "num_targets_with_at_least_one_candidate": 0,
        "min_candidates_per_target": 0,
        "max_candidates_per_target": 0,
        "avg_candidates_per_target": 0.0,
        "median_candidates_per_target": 0.0,
        "failures_by_code": {"adapter.route_invalid": 1, "adapter.schema_invalid": 2},
    }


def test_collected_statistics_include_target_distribution_and_failures() -> None:
    statistics = collected_candidate_statistics(
        {
            "target-a": [failed_candidate(1, "adapter.schema_invalid")],
            "target-b": [
                failed_candidate(1, "adapter.route_invalid"),
                failed_candidate(2, "adapter.route_invalid"),
            ],
            "target-c": [],
        }
    )

    assert statistics.targets_with_at_least_one_candidate == {"target-a", "target-b"}
    assert statistics.failures_by_target == {
        "target-a": {"adapter.schema_invalid": 1},
        "target-b": {"adapter.route_invalid": 2},
    }
    assert statistics.min_candidates_per_target == 1
    assert statistics.max_candidates_per_target == 2
    assert statistics.avg_candidates_per_target == 1.5
    assert statistics.median_candidates_per_target == 1.5
