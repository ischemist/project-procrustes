from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from retrocast.exceptions import RetroCastException
from retrocast.typing import InchiKeyStr, SmilesStr

CheckStatus = Literal["pass", "fail", "unknown", "not_evaluated"]
ValidityTier: TypeAlias = int
ScopeId: TypeAlias = Literal["stock"]
SUPPORTED_VALIDITY_TIERS = frozenset({0, 1, 2, 3})
IMPLEMENTED_VALIDITY_TIERS = frozenset({0})


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
    status: CheckStatus
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TierResult(BaseModel):
    tier: ValidityTier
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ReactionValidity(BaseModel):
    reaction_index: int
    product_smiles: SmilesStr
    product_inchikey: InchiKeyStr
    reactant_smiles: list[SmilesStr] = Field(default_factory=list)
    reactant_inchikeys: list[InchiKeyStr] = Field(default_factory=list)
    tiers: dict[int, TierResult] = Field(default_factory=dict)

    def satisfies_validity(self, tier: ValidityTier = 0) -> bool:
        return self.tiers.get(tier, TierResult(tier=tier, status="unknown")).status == "pass"


class RouteValidity(BaseModel):
    tiers: dict[int, TierResult] = Field(default_factory=dict)
    reactions: list[ReactionValidity] = Field(default_factory=list)

    def satisfies_validity(self, tier: ValidityTier = 0) -> bool:
        return self.tiers.get(tier, TierResult(tier=tier, status="unknown")).status == "pass"


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
