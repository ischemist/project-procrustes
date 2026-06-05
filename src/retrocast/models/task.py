from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

from retrocast.chem import get_inchi_key
from retrocast.exceptions import ChemError
from retrocast.models.route import Route
from retrocast.typing import InChIKeyStr, SmilesStr

STOCK_TERMINATION = "retrocast.stock_termination"
REQUIRED_LEAVES = "retrocast.required_leaves"
ROUTE_DEPTH = "retrocast.route_depth"


class Target(BaseModel):
    id: str
    smiles: SmilesStr
    inchikey: InChIKeyStr
    acceptable_routes: list[Route] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity(self) -> Target:
        try:
            expected_inchikey = get_inchi_key(self.smiles)
        except ChemError as exc:
            raise ValueError("Target.smiles must be valid.") from exc
        if self.inchikey != expected_inchikey:
            raise ValueError("Target.inchikey must match Target.smiles.")
        return self


class TaskConstraint(BaseModel):
    kind: str
    model_config = ConfigDict(extra="allow")


class StockTerminationConstraint(TaskConstraint):
    kind: Literal["retrocast.stock_termination"] = STOCK_TERMINATION
    stock: str


class RequiredLeavesConstraint(TaskConstraint):
    kind: Literal["retrocast.required_leaves"] = REQUIRED_LEAVES
    smiles: list[SmilesStr]


class RouteDepthConstraint(TaskConstraint):
    kind: Literal["retrocast.route_depth"] = ROUTE_DEPTH
    max_depth: int | Literal["short", "medium", "long"]


_KNOWN_CONSTRAINTS: dict[str, type[TaskConstraint]] = {
    STOCK_TERMINATION: StockTerminationConstraint,
    REQUIRED_LEAVES: RequiredLeavesConstraint,
    ROUTE_DEPTH: RouteDepthConstraint,
}


def _hydrate_task_constraints(value: object) -> list[TaskConstraint]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"task constraints must be a list, got {type(value).__name__}")

    constraints = []
    for item in value:
        if isinstance(item, TaskConstraint):
            data: dict[str, Any] = item.model_dump(mode="python")
        elif isinstance(item, dict):
            data = {str(key): val for key, val in item.items()}
        else:
            raise TypeError(f"unsupported task constraint: {type(item).__name__}")
        kind = data.get("kind")
        model = _KNOWN_CONSTRAINTS.get(kind) if isinstance(kind, str) else None
        constraints.append((model or TaskConstraint).model_validate(data))
    return constraints


class Task(BaseModel):
    name: str
    targets: dict[str, Target]
    # Pydantic serializes fields through their annotation. Without SerializeAsAny,
    # subclass payload fields like stock/smiles/max_depth are dropped.
    default_constraints: list[SerializeAsAny[TaskConstraint]] = Field(default_factory=list)
    constraints: dict[str, list[SerializeAsAny[TaskConstraint]]] = Field(default_factory=dict)
    metric_label: str | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"

    @field_validator("default_constraints", mode="before")
    @classmethod
    def _parse_default_constraints(cls, value: object) -> list[TaskConstraint]:
        return _hydrate_task_constraints(value)

    @field_validator("constraints", mode="before")
    @classmethod
    def _parse_target_constraints(cls, value: object) -> dict[str, list[TaskConstraint]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"Task.constraints must be a dict, got {type(value).__name__}")
        return {str(target_id): _hydrate_task_constraints(constraints) for target_id, constraints in value.items()}

    @model_validator(mode="after")
    def _validate_target_keys(self) -> Task:
        mismatches = [key for key, target in self.targets.items() if key != target.id]
        if mismatches:
            raise ValueError("Task.targets keys must match Target.id values.")
        _validate_unique_constraint_kinds("Task.default_constraints", self.default_constraints)
        for target_id, constraints in self.constraints.items():
            _validate_unique_constraint_kinds(f"Task.constraints[{target_id!r}]", constraints)
        return self

    def effective_constraints(self, target_id: str) -> list[TaskConstraint]:
        by_kind = {constraint.kind: constraint for constraint in self.default_constraints}
        by_kind.update({constraint.kind: constraint for constraint in self.constraints.get(target_id, [])})
        return list(by_kind.values())

    def derived_metric_label(self) -> str:
        if self.metric_label is not None:
            return self.metric_label

        stocks = set()
        has_leaf = False
        has_depth = False
        for target_id in self.targets:
            for constraint in self.effective_constraints(target_id):
                payload = constraint.model_dump(mode="python", exclude_none=True)
                if constraint.kind == STOCK_TERMINATION and payload.get("stock") is not None:
                    stocks.add(payload["stock"])
                elif constraint.kind == REQUIRED_LEAVES:
                    has_leaf = True
                elif constraint.kind == ROUTE_DEPTH:
                    has_depth = True

        parts = []
        if len(stocks) == 1:
            parts.append(next(iter(stocks)))
        elif len(stocks) > 1:
            parts.append("stocks")
        if has_leaf:
            parts.append("leaf")
        if has_depth:
            parts.append("depth")
        return "+".join(parts) if parts else "task"


class Benchmark(Task):
    description: str = ""


def _validate_unique_constraint_kinds(label: str, constraints: list[TaskConstraint]) -> None:
    seen = set()
    duplicates = []
    for constraint in constraints:
        if constraint.kind in seen:
            duplicates.append(constraint.kind)
        seen.add(constraint.kind)
    if duplicates:
        raise ValueError(f"{label} contains duplicate constraint kinds: {sorted(set(duplicates))}")
