from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from retrocast.models.chem import Route
from retrocast.models.validity import FailureRecord


class CandidateSource(BaseModel):
    key: str | None = None
    row_index: int | None = None
    record_id: str | None = None


class CandidateRecord(BaseModel):
    rank: int
    route: Route | None = None
    adapter_failure: FailureRecord | None = None
    source: CandidateSource = Field(default_factory=CandidateSource)

    @model_validator(mode="after")
    def _require_route_or_failure(self) -> CandidateRecord:
        if self.route is None and self.adapter_failure is None:
            raise ValueError("CandidateRecord requires either route or adapter_failure.")
        if self.route is not None and self.adapter_failure is not None:
            raise ValueError("CandidateRecord cannot contain both route and adapter_failure.")
        return self


CandidateRecordsDict = dict[str, list[CandidateRecord]]


class CandidateAuditMetadata(BaseModel):
    candidate_audit_version: str = "1"
    preserves_failed_candidates: bool = False
    candidate_denominator: Literal["complete", "partial", "route_only"] = "route_only"
    target_assignment: Literal["benchmark_target_key", "target_hint", "route_match", "unknown"] = "unknown"
    n_raw_entries_seen: int = 0
    n_candidate_records_written: int = 0
    n_routes_adapted: int = 0
    n_adaptation_failures: int = 0
    n_unassigned_candidates: int = 0
    sampling_policy: str | None = None


class CandidateRecordsArtifact(BaseModel):
    metadata: CandidateAuditMetadata = Field(default_factory=CandidateAuditMetadata)
    records: CandidateRecordsDict = Field(default_factory=dict)
