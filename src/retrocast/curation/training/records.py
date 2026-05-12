from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import BaseModel
from pydantic import Field as PydanticField

from retrocast.models.chem import ReactionSignature, Route
from retrocast.typing import ReactionSmilesStr, SmilesStr

TrainingHoldoutMode = Literal["route", "reaction"]
SplitName = Literal["training", "validation"]
ConditionIdentity = tuple[str, ...] | str | None
ReactionIdentityKey = tuple[tuple[SmilesStr, ...], SmilesStr, ConditionIdentity]


@dataclass(frozen=True)
class RawRouteSource:
    dataset: str
    raw_index: int
    raw_route_hash: str
    patent_id: str | None = None


@dataclass
class TrainingRouteRecord:
    id: str
    split: SplitName
    route_signature: str
    content_hash: str
    route: Route
    sources: list[RawRouteSource] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "route_signature": self.route_signature,
            "content_hash": self.content_hash,
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
        source_payload = payload.get("source")
        if not isinstance(source_payload, Mapping):
            raise ValueError("training route record is missing source metadata")

        dataset = source_payload.get("dataset")
        raw_indices = source_payload.get("raw_indices")
        raw_route_hashes = source_payload.get("raw_route_hashes")
        patent_ids = source_payload.get("patent_ids")
        route_payload = payload.get("route")
        split = payload.get("split")
        record_id = payload.get("id")
        route_signature = payload.get("route_signature")
        content_hash = payload.get("content_hash")

        if not isinstance(dataset, str):
            raise ValueError("training route record source dataset must be a string")
        if split not in ("training", "validation"):
            raise ValueError("training route record split must be 'training' or 'validation'")
        if not isinstance(record_id, str):
            raise ValueError("training route record id must be a string")
        if not isinstance(route_signature, str):
            raise ValueError("training route record route_signature must be a string")
        if not isinstance(content_hash, str):
            raise ValueError("training route record content_hash must be a string")
        if not isinstance(raw_indices, list) or not all(isinstance(value, int) for value in raw_indices):
            raise ValueError("training route record raw_indices must be a list of ints")
        if not isinstance(raw_route_hashes, list) or not all(isinstance(value, str) for value in raw_route_hashes):
            raise ValueError("training route record raw_route_hashes must be a list of strings")
        if not isinstance(patent_ids, list) or not all(value is None or isinstance(value, str) for value in patent_ids):
            raise ValueError("training route record patent_ids must be a list of strings or nulls")
        if len(raw_indices) != len(raw_route_hashes) or len(raw_indices) != len(patent_ids):
            raise ValueError("training route record source arrays must have equal lengths")
        if not isinstance(route_payload, Mapping):
            raise ValueError("training route record is missing route payload")

        typed_raw_indices = cast(list[int], raw_indices)
        typed_raw_route_hashes = cast(list[str], raw_route_hashes)
        typed_patent_ids = cast(list[str | None], patent_ids)

        return cls(
            id=record_id,
            split=split,
            route_signature=route_signature,
            content_hash=content_hash,
            route=Route.model_validate(route_payload),
            sources=[
                RawRouteSource(
                    dataset=dataset,
                    raw_index=raw_index,
                    raw_route_hash=raw_route_hash,
                    patent_id=patent_id,
                )
                for raw_index, raw_route_hash, patent_id in zip(
                    typed_raw_indices, typed_raw_route_hashes, typed_patent_ids, strict=True
                )
            ],
        )


@dataclass(frozen=True)
class TrainingSetBuildConfig:
    holdout_mode: TrainingHoldoutMode
    val_fraction: float = 0.05
    seed: int = 20260502
    route_prefix: str = "paroutes"
    show_progress: bool = True

    @property
    def release_name(self) -> str:
        return f"{self.holdout_mode}-heldout-n1-n5"

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
                "transform_dedup": "collapse mapped-smiles variants by structure + condition identity + raw paroutes reaction_hash sidecar",
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
    transform_ids_by_source_id: dict[str, str]


@dataclass
class PreparedTrainingRoute:
    route: Route
    structural_signature: str
    sources: list[RawRouteSource] = field(default_factory=list)
    transform_ids_by_source_id: dict[str, str] = field(default_factory=dict)


class TrainingReactionSource(BaseModel):
    route_id: str
    step_index: int
    source_id: str | None = None
    dataset: str
    raw_route_indices: list[int] = PydanticField(default_factory=list)
    raw_route_hashes: list[str] = PydanticField(default_factory=list)
    patent_ids: list[str | None] = PydanticField(default_factory=list)


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
class PreparedTrainingReaction:
    reactants: tuple[SmilesStr, ...]
    product: SmilesStr
    mapped_smiles: ReactionSmilesStr
    alternative_mapped_smiles: list[ReactionSmilesStr] = field(default_factory=list)
    condition_slot: str | None = None
    condition_slot_smiles: tuple[SmilesStr, ...] = field(default_factory=tuple)
    transform_id: str | None = None
    sources: list[TrainingReactionSource] = field(default_factory=list)


@dataclass
class TrainingReactionBuildResult:
    release_name: str
    records: list[TrainingReactionRecord]
    summary: dict[str, Any]


@dataclass(frozen=True)
class NonFatalConditionSlotParseStatistics:
    malformed_rsmi_count: int = 0
    uncanonicalizable_token_count: int = 0
    distinct_uncanonicalizable_token_count: int = 0

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "malformed_rsmi_count": self.malformed_rsmi_count,
            "uncanonicalizable_token_count": self.uncanonicalizable_token_count,
            "distinct_uncanonicalizable_token_count": self.distinct_uncanonicalizable_token_count,
        }


@dataclass(frozen=True)
class AdaptationStatistics:
    raw_routes: int
    adapted_routes: int
    skipped_routes: int
    skipped_without_error_code: int
    failures_by_code: dict[str, int]
    non_fatal_condition_slot_parse: NonFatalConditionSlotParseStatistics | None = None

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
        return manifest_dict
