from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.models.route import Route


class Target(BaseModel):
    id: str
    smiles: SmilesStr
    inchikey: InChIKeyStr
    acceptable_routes: list[Route] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)


class TaskConstraints(BaseModel):
    stock: str | None = None
    required_leaves_smiles: list[SmilesStr] | None = None
    route_depth: int | Literal["short", "medium", "long"] | None = None


class Task(BaseModel):
    name: str
    targets: dict[str, Target]
    default_constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    constraints: dict[str, TaskConstraints] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"


class Benchmark(Task):
    description: str = ""
