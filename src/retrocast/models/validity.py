from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_serializer, model_validator

from retrocast.exceptions import RetroCastException
from retrocast.typing import InchiKeyStr, SmilesStr

CheckStatus = Literal["pass", "fail", "unknown", "not_evaluated"]
ValidityTier: TypeAlias = int
ScopeId: TypeAlias = Literal["stock"]
SUPPORTED_VALIDITY_TIERS = frozenset({0, 1, 2, 3})
IMPLEMENTED_VALIDITY_TIERS = frozenset({0})


def tier_key(tier: ValidityTier) -> str:
    return f"tier {tier}"


def _parse_tier_key(key: Any) -> ValidityTier | None:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        if key.isdecimal():
            return int(key)
        prefix = "tier "
        if key.startswith(prefix) and key[len(prefix) :].isdecimal():
            return int(key[len(prefix) :])
    return None


def _deserialize_tiers(data: dict[str, Any]) -> dict[int, TierResult]:
    raw_tiers = data.get("tiers", {})
    tier_items: dict[Any, Any] = dict(raw_tiers) if isinstance(raw_tiers, dict) else {}
    tier_items.update({key: value for key, value in data.items() if _parse_tier_key(key) is not None})
    tiers: dict[int, TierResult] = {}
    for key, value in tier_items.items():
        tier = _parse_tier_key(key)
        if tier is None or not isinstance(value, dict):
            continue
        tiers[tier] = TierResult.model_validate({"tier": tier, **value})
    return tiers


def _serialize_tiers(tiers: dict[int, TierResult]) -> dict[str, Any]:
    return {tier_key(tier): result.to_artifact_dict() for tier, result in sorted(tiers.items())}


class FailureRecord(BaseModel):
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False

    @classmethod
    def from_exception(cls, error: RetroCastException) -> FailureRecord:
        return cls.model_validate(error.to_dict())


class CheckResult(BaseModel):
    code: str
    status: CheckStatus = "fail"
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="plain")
    def to_artifact_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code}
        if self.status != "fail":
            data["status"] = self.status
        if self.message is not None:
            data["message"] = self.message
        if self.details:
            data["details"] = self.details
        return data


class TierResult(BaseModel):
    tier: ValidityTier
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)

    @model_serializer(mode="plain")
    def to_artifact_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status}
        if self.checks:
            data["checks"] = self.checks
        return data


class ReactionValidity(BaseModel):
    reaction_id: str
    tiers: dict[int, TierResult] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_artifact_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        validity = data.get("validity")
        if isinstance(validity, dict):
            data = {**data, "tiers": _deserialize_tiers(validity)}
        return data

    def satisfies_validity(self, tier: ValidityTier = 0) -> bool:
        return self.tiers.get(tier, TierResult(tier=tier, status="unknown")).status == "pass"

    @model_serializer(mode="plain")
    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "validity": _serialize_tiers(self.tiers),
        }


class RouteValidity(BaseModel):
    tiers: dict[int, TierResult] = Field(default_factory=dict)
    reactions: list[ReactionValidity] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_artifact_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "tiers" not in data:
            data = {**data, "tiers": _deserialize_tiers(data)}
        return data

    def satisfies_validity(self, tier: ValidityTier = 0) -> bool:
        return self.tiers.get(tier, TierResult(tier=tier, status="unknown")).status == "pass"

    @model_serializer(mode="plain")
    def to_artifact_dict(self) -> dict[str, Any]:
        data = _serialize_tiers(self.tiers)
        if self.reactions:
            data["reactions"] = self.reactions
        return data


class StockTerminationConstraint(BaseModel):
    type: Literal["stock_termination"] = "stock_termination"
    stock_name: str
    match_level: str


class RequiredLeafConstraint(BaseModel):
    type: Literal["required_leaf"] = "required_leaf"
    smiles: SmilesStr | None = None
    inchikey: InchiKeyStr | None = None
    match_level: str

    @model_validator(mode="after")
    def _require_leaf_identity(self) -> RequiredLeafConstraint:
        if self.smiles is None and self.inchikey is None:
            raise ValueError("RequiredLeafConstraint requires smiles or inchikey.")
        return self


class MaxDepthConstraint(BaseModel):
    type: Literal["max_depth"] = "max_depth"
    max_depth: int


Constraint = Annotated[
    StockTerminationConstraint | RequiredLeafConstraint | MaxDepthConstraint,
    Field(discriminator="type"),
]


class MetricScope(BaseModel):
    id: str
    constraints: list[Constraint] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)
