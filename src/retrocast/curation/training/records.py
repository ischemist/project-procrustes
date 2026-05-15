from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from retrocast.adapters.paroutes_diagnostics import ConditionSlotParseStatistics, PatentIdParseStatistics
from retrocast.models.chem import ReactionSignature, Route
from retrocast.typing import ReactionSmilesStr, SmilesStr

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["training", "validation"]
ConditionIdentity = tuple[str, ...] | str | None
ReactionIdentityKey = tuple[tuple[SmilesStr, ...], SmilesStr, ConditionIdentity]


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
    sources: list[RawRouteSource] = PydanticField(default_factory=list)

    @property
    def route_signature(self) -> str:
        return self.route.get_structural_signature()

    @model_validator(mode="before")
    @classmethod
    def _accept_release_source_shape(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "sources" in data or "source" not in data:
            return data

        source = data["source"]
        if not isinstance(source, Mapping):
            return data

        raw_indices = source.get("raw_indices")
        raw_route_hashes = source.get("raw_route_hashes")
        patent_ids = source.get("patent_ids")
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            return data
        if not isinstance(raw_route_hashes, Sequence) or isinstance(raw_route_hashes, (str, bytes)):
            return data
        if not isinstance(patent_ids, Sequence) or isinstance(patent_ids, (str, bytes)):
            return data

        return {
            **data,
            "sources": [
                {
                    "dataset": source.get("dataset"),
                    "raw_index": raw_index,
                    "raw_route_hash": raw_route_hash,
                    "patent_id": patent_id,
                }
                for raw_index, raw_route_hash, patent_id in zip(raw_indices, raw_route_hashes, patent_ids, strict=True)
            ],
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "source": {
                "dataset": self.sources[0].dataset if self.sources else "unknown",
                "raw_indices": [source.raw_index for source in self.sources],
                "raw_route_hashes": [source.raw_route_hash for source in self.sources],
                "patent_ids": [source.patent_id for source in self.sources],
            },
            "route": self.route.model_dump(mode="json"),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> TrainingRouteRecord:
        return cls.model_validate(payload)


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
            "split": {
                "val_fraction": self.val_fraction,
                "seed": self.seed,
            },
            "progress": {
                "enabled": self.show_progress,
            },
            "release_rules": {
                "route_holdout": "exclude full n1 union n5 by route.get_structural_signature()",
                "reaction_holdout": "excise reactions present in n1 union n5 and keep surviving sub-routes",
                "chemical_exact_dedup": "collapse exact route duplicates by route.get_annotated_signature(include_mapped_smiles=True)",
                "transform_dedup": "collapse mapped-smiles variants by structure + condition identity",
            },
        }


@dataclass
class TrainingSetBuildResult:
    release_name: str
    records: list[TrainingRouteRecord]
    summary: dict[str, Any]


@dataclass(frozen=True)
class AdaptedTrainingRoute:
    route: Route
    structural_signature: str
    reaction_signatures: set[ReactionSignature]
    source: RawRouteSource


@dataclass
class PreparedTrainingRoute:
    route: Route
    structural_signature: str
    sources: list[RawRouteSource] = field(default_factory=list)


class TrainingReactionSource(BaseModel):
    route_id: str
    step_index: int
    source_id: str | None = None


class TrainingReactionRecord(BaseModel):
    id: str
    split: SplitName
    reactants: list[SmilesStr]
    product: SmilesStr
    mapped_smiles: ReactionSmilesStr
    alternative_mapped_smiles: list[ReactionSmilesStr] = PydanticField(default_factory=list)
    condition_slot: str | None = None
    condition_slot_smiles: list[SmilesStr] = PydanticField(default_factory=list)
    sources: list[TrainingReactionSource] = PydanticField(default_factory=list)

    def to_rsmi_line(self) -> str:
        return self.mapped_smiles


@dataclass
class TrainingReactionCandidate:
    reactants: tuple[SmilesStr, ...]
    product: SmilesStr
    mapped_smiles: ReactionSmilesStr
    alternative_mapped_smiles: list[ReactionSmilesStr] = field(default_factory=list)
    condition_slot: str | None = None
    condition_slot_smiles: tuple[SmilesStr, ...] = field(default_factory=tuple)
    sources: list[TrainingReactionSource] = field(default_factory=list)


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
    patent_id_parse: PatentIdParseStatistics | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        manifest_dict = {
            "raw_routes": self.raw_routes,
            "adapted_routes": self.adapted_routes,
            "skipped_routes": self.skipped_routes,
            "skipped_without_error_code": self.skipped_without_error_code,
            "failures_by_code": dict(sorted(self.failures_by_code.items())),
        }
        if self.non_fatal_condition_slot_parse is not None:
            manifest_dict["non_fatal_condition_slot_parse"] = self.non_fatal_condition_slot_parse.to_manifest_dict()
        if self.patent_id_parse is not None:
            manifest_dict["patent_id_parse"] = self.patent_id_parse.to_manifest_dict()
        return manifest_dict
