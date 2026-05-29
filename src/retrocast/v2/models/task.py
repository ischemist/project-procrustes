from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from retrocast.chem import get_inchi_key
from retrocast.exceptions import ChemError
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.models.route import Route


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


class TaskConstraints(BaseModel):
    stock: str | None = None
    required_leaves_smiles: list[SmilesStr] | None = None
    route_depth: int | Literal["short", "medium", "long"] | None = None

    @cached_property
    def required_leaf_inchikeys(self) -> tuple[InChIKeyStr, ...]:
        if not self.required_leaves_smiles:
            return ()
        return tuple(get_inchi_key(smiles) for smiles in self.required_leaves_smiles)


class Task(BaseModel):
    name: str
    targets: dict[str, Target]
    default_constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    constraints: dict[str, TaskConstraints] = Field(default_factory=dict)
    metric_label: str | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"

    @model_validator(mode="after")
    def _validate_target_keys(self) -> Task:
        mismatches = [key for key, target in self.targets.items() if key != target.id]
        if mismatches:
            raise ValueError("Task.targets keys must match Target.id values.")
        return self

    def effective_constraints(self, target_id: str) -> TaskConstraints:
        return self.constraints.get(target_id, self.default_constraints)

    def derived_metric_label(self) -> str:
        if self.metric_label is not None:
            return self.metric_label

        stocks = set()
        has_leaf = False
        has_depth = False
        for target_id in self.targets:
            constraints = self.effective_constraints(target_id)
            if constraints.stock is not None:
                stocks.add(constraints.stock)
            if constraints.required_leaves_smiles:
                has_leaf = True
            if constraints.route_depth is not None:
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
