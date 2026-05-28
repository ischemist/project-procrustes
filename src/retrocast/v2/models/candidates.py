from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from retrocast.typing import ErrorCode, InChIKeyStr, SmilesStr
from retrocast.v2.models.route import Route


class FailureRecord(BaseModel):
    code: ErrorCode
    message: str | None = None
    target_id: str | None = None
    target_smiles: SmilesStr | None = None
    target_inchikey: InChIKeyStr | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    rank: int
    route: Route | None = None
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _require_route_or_failure(self) -> Candidate:
        if self.route is None and self.failure is None:
            raise ValueError("Candidate requires route or failure.")
        if self.route is not None and self.failure is not None:
            raise ValueError("Candidate cannot contain both route and failure.")
        return self
