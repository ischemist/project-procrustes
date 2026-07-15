from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel, Field

from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import Evaluation
from retrocast.workflow.collect import NativeCollectedCandidates


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
        return cast(int, self._manifest()["min_candidates_per_target"])

    @property
    def max_candidates_per_target(self) -> int:
        return cast(int, self._manifest()["max_candidates_per_target"])

    @property
    def avg_candidates_per_target(self) -> float:
        return cast(float, self._manifest()["avg_candidates_per_target"])

    @property
    def median_candidates_per_target(self) -> float:
        return cast(float, self._manifest()["median_candidates_per_target"])

    def to_manifest_dict(self) -> dict[str, object]:
        return self._manifest()

    def _manifest(self) -> dict[str, object]:
        from retrocast import _native

        payload = self.model_dump_json(exclude_none=True)
        return dict(json.loads(_native.candidate_run_manifest_json(payload)))


def candidate_statistics(candidates: Sequence[Candidate]) -> CandidateRunStatistics:
    from retrocast import _native

    payload = json.dumps(
        [candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates],
        separators=(",", ":"),
    )
    return CandidateRunStatistics.model_validate_json(_native.candidate_statistics_json(payload))


def collected_candidate_statistics(candidates_by_target: Mapping[str, Sequence[Candidate]]) -> CandidateRunStatistics:
    from retrocast import _native

    handle = (
        candidates_by_target.native_handle() if isinstance(candidates_by_target, NativeCollectedCandidates) else None
    )
    if handle is not None:
        payload = _native.collected_candidate_statistics_native(handle)
    else:
        predictions = {
            target_id: [candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates]
            for target_id, candidates in candidates_by_target.items()
        }
        payload = _native.collected_candidate_statistics_json(json.dumps(predictions, separators=(",", ":")))
    return CandidateRunStatistics.model_validate_json(payload)


def evaluation_statistics(evaluation: Evaluation) -> dict[str, object]:
    from retrocast import _native
    from retrocast.native import NativeEvaluation

    handle = evaluation.native_handle() if isinstance(evaluation, NativeEvaluation) else None
    if handle is not None:
        payload = _native.evaluation_statistics_native(handle)
    else:
        payload = _native.evaluation_statistics_json(evaluation.model_dump_json(exclude_none=True))
    return dict(json.loads(payload))
