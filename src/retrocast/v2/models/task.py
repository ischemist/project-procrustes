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
    annotations: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"

    @model_validator(mode="after")
    def _validate_target_keys(self) -> Task:
        mismatches = [key for key, target in self.targets.items() if key != target.id]
        if mismatches:
            raise ValueError("Task.targets keys must match Target.id values.")
        return self


class Benchmark(Task):
    description: str = ""
