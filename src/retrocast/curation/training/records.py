from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from retrocast.adapters.paroutes import ConditionSlotParseStatistics
from retrocast.models.route import Route
from retrocast.typing import ReactionSmilesStr, SmilesStr

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["training", "validation"]


class RawRouteSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    raw_index: int
    raw_route_hash: str
    patent_id: str | None = None


class TrainingRouteRecord(BaseModel):
    id: str
    split: SplitName
    route: Route
    sources: list[RawRouteSource] = Field(default_factory=list)

    @property
    def route_signature(self) -> str:
        return self.route.signature()


class TrainingReactionSource(BaseModel):
    route_id: str
    step_index: int | None = None
    reaction_id: str
    source_id: str | None = None


class TrainingReactionRecord(BaseModel):
    id: str
    split: SplitName
    reactants: list[SmilesStr]
    product: SmilesStr
    mapped_smiles: ReactionSmilesStr
    alternative_mapped_smiles: list[ReactionSmilesStr] = Field(default_factory=list)
    condition_slot: str | None = None
    condition_slot_smiles: list[SmilesStr] = Field(default_factory=list)
    sources: list[TrainingReactionSource] = Field(default_factory=list)

    def to_rsmi_line(self) -> str:
        return self.mapped_smiles


@dataclass(frozen=True)
class TrainingSetBuildConfig:
    holdout_mode: TrainingHoldoutMode
    val_fraction: float = 0.05
    seed: int = 20260502
    route_prefix: str = "paroutes"
    show_progress: bool = True

    @property
    def release_name(self) -> str:
        return f"{self.holdout_mode}-holdout-n1-n5"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "holdout_mode": self.holdout_mode,
            "split": {"val_fraction": self.val_fraction, "seed": self.seed},
            "progress": {"enabled": self.show_progress},
            "release_rules": {
                "route_holdout": "exclude full n1 union n5 by route.signature()",
                "reaction_holdout": "excise reactions present in n1 union n5 and keep surviving sub-routes",
                "chemical_exact_dedup": "collapse exact route duplicates by structure, mapped reactions, and condition identity",
                "transform_dedup": "collapse mapped-smiles variants by structure + condition identity",
            },
        }


@dataclass(frozen=True)
class AdaptedTrainingRoute:
    route: Route
    source: RawRouteSource


@dataclass(frozen=True)
class TrainingRouteAdaptation:
    routes: list[AdaptedTrainingRoute]
    stats: AdaptationStatistics


@dataclass
class PreparedTrainingRoute:
    route: Route
    structural_signature: str
    sources: list[RawRouteSource] = field(default_factory=list)


@dataclass
class TrainingSetBuildResult:
    release_name: str
    records: list[TrainingRouteRecord]
    summary: dict[str, Any]


@dataclass
class TrainingReactionBuildResult:
    release_name: str
    records: list[TrainingReactionRecord]
    summary: dict[str, Any]


class AdaptationStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_routes: int
    adapted_routes: int
    skipped_routes: int
    skipped_without_error_code: int
    failures_by_code: dict[str, int]
    non_fatal_condition_slot_parse: ConditionSlotParseStatistics | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = {
            "raw_routes": self.raw_routes,
            "adapted_routes": self.adapted_routes,
            "skipped_routes": self.skipped_routes,
            "skipped_without_error_code": self.skipped_without_error_code,
            "failures_by_code": dict(sorted(self.failures_by_code.items())),
        }
        if self.non_fatal_condition_slot_parse is not None:
            payload["non_fatal_condition_slot_parse"] = self.non_fatal_condition_slot_parse.model_dump(mode="json")
        return payload
