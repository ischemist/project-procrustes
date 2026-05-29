from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import Evaluation, Tier


class CandidateRunStatistics(BaseModel):
    total_candidates_seen: int = 0
    successful_candidates: int = 0
    failed_candidates: int = 0
    final_candidates_saved: int = 0
    targets_with_at_least_one_candidate: set[str] = Field(default_factory=set)
    candidates_per_target: dict[str, int] = Field(default_factory=dict)
    failures_by_code: dict[str, int] = Field(default_factory=dict)
    failures_by_target: dict[str, dict[str, int]] = Field(default_factory=dict)

    @property
    def num_targets_with_candidates(self) -> int:
        return len(self.targets_with_at_least_one_candidate)

    @property
    def min_candidates_per_target(self) -> int:
        return min(self.candidates_per_target.values()) if self.candidates_per_target else 0

    @property
    def max_candidates_per_target(self) -> int:
        return max(self.candidates_per_target.values()) if self.candidates_per_target else 0

    @property
    def avg_candidates_per_target(self) -> float:
        if not self.candidates_per_target:
            return 0.0
        return round(statistics.mean(self.candidates_per_target.values()), 2)

    @property
    def median_candidates_per_target(self) -> float:
        if not self.candidates_per_target:
            return 0.0
        return round(statistics.median(self.candidates_per_target.values()), 2)

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "total_candidates_seen": self.total_candidates_seen,
            "successful_candidates": self.successful_candidates,
            "failed_candidates": self.failed_candidates,
            "final_candidates_saved": self.final_candidates_saved,
            "num_targets_with_at_least_one_candidate": self.num_targets_with_candidates,
            "min_candidates_per_target": self.min_candidates_per_target,
            "max_candidates_per_target": self.max_candidates_per_target,
            "avg_candidates_per_target": self.avg_candidates_per_target,
            "median_candidates_per_target": self.median_candidates_per_target,
            "failures_by_code": dict(sorted(self.failures_by_code.items())),
        }


def candidate_statistics(candidates: Sequence[Candidate]) -> CandidateRunStatistics:
    stats = CandidateRunStatistics(total_candidates_seen=len(candidates), final_candidates_saved=len(candidates))
    for candidate in candidates:
        if candidate.failure is None:
            stats.successful_candidates += 1
        else:
            stats.failed_candidates += 1
            code = str(candidate.failure.code)
            stats.failures_by_code[code] = stats.failures_by_code.get(code, 0) + 1
    return stats


def collected_candidate_statistics(candidates_by_target: Mapping[str, Sequence[Candidate]]) -> CandidateRunStatistics:
    flat_candidates = [candidate for candidates in candidates_by_target.values() for candidate in candidates]
    stats = candidate_statistics(flat_candidates)
    for target_id, candidates in candidates_by_target.items():
        if candidates:
            stats.targets_with_at_least_one_candidate.add(target_id)
            stats.candidates_per_target[target_id] = len(candidates)
        for candidate in candidates:
            if candidate.failure is None:
                continue
            code = str(candidate.failure.code)
            target_failures = stats.failures_by_target.setdefault(target_id, {})
            target_failures[code] = target_failures.get(code, 0) + 1
    return stats


def evaluation_statistics(evaluation: Evaluation) -> dict[str, object]:
    candidates = [candidate for target in evaluation.targets.values() for candidate in target.candidates]
    stats: dict[str, object] = {
        "n_targets": len(evaluation.targets),
        "n_candidates": len(candidates),
        "n_failed_candidates": sum(1 for candidate in candidates if candidate.failed_adaptation()),
    }
    for tier in evaluation.tiers:
        key = int(tier)
        stats[f"n_solv_{key}"] = sum(
            1
            for target in evaluation.targets.values()
            if any(candidate.satisfies_solv(Tier(key)) for candidate in target.candidates)
        )
    return stats
