from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retrocast.curation.training.records import (
    TrainingReactionBuildResult,
    TrainingReactionRecord,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
)
from retrocast.exceptions import TrainingReleaseError
from retrocast.hashing import hash_json
from retrocast.io import ContentType, create_manifest, save_jsonl_gz, save_lines_gz

TRAINING_REACTION_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-single-step-release"


class TrainingReactionReleaseBuilder:
    def __init__(self, *, route_records: Sequence[TrainingRouteRecord], config: TrainingSetBuildConfig) -> None:
        self.route_records = route_records
        self.config = config
        self._started = False

    def build(self) -> TrainingReactionBuildResult:
        if self._started:
            raise RuntimeError("TrainingReactionReleaseBuilder instances are single-use")
        self._started = True

        from retrocast import native

        try:
            payload = native.build_training_reaction_release(self.route_records, self.config)
        except native.NativeTrainingError as error:
            raise TrainingReleaseError(
                str(error.payload.get("message", error)),
                code=str(error.payload.get("code", "workflow.training_release_error")),
                context=error.payload.get("context") if isinstance(error.payload.get("context"), dict) else {},
            ) from error
        for message in payload.pop("_warnings", []):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return TrainingReactionBuildResult(
            release_name=payload["release_name"],
            records=[TrainingReactionRecord.model_validate(record) for record in payload["records"]],
            summary=payload["summary"],
        )


def write_training_reaction_release(
    *,
    result: TrainingReactionBuildResult,
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
    config: TrainingSetBuildConfig,
) -> None:
    release_dir = output_dir / result.release_name
    all_path = release_dir / "all.jsonl.gz"
    training_path = release_dir / "training.jsonl.gz"
    validation_path = release_dir / "validation.jsonl.gz"
    all_rsmi_path = release_dir / "all.rsmi.txt.gz"
    training_rsmi_path = release_dir / "training.rsmi.txt.gz"
    validation_rsmi_path = release_dir / "validation.rsmi.txt.gz"
    manifest_path = release_dir / "manifest.json"
    training = [record for record in result.records if record.split == "training"]
    validation = [record for record in result.records if record.split == "validation"]
    save_jsonl_gz(result.records, all_path)
    save_jsonl_gz(training, training_path)
    save_jsonl_gz(validation, validation_path)
    save_lines_gz((record.to_rsmi_line() for record in result.records), all_rsmi_path)
    save_lines_gz((record.to_rsmi_line() for record in training), training_rsmi_path)
    save_lines_gz((record.to_rsmi_line() for record in validation), validation_rsmi_path)
    manifest = create_manifest(
        action=TRAINING_REACTION_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            ("all", all_path, result.records, ContentType.UNKNOWN, reaction_records_content_hash(result.records)),
            ("training", training_path, training, ContentType.UNKNOWN, reaction_records_content_hash(training)),
            ("validation", validation_path, validation, ContentType.UNKNOWN, reaction_records_content_hash(validation)),
        ],
        root_dir=source_root,
        parameters=config.to_manifest_dict(),
        statistics=result.summary.get("output", {}),
        summary=result.summary,
        release_name=result.release_name,
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def summarize_reaction_records(records: Sequence[TrainingReactionRecord]) -> dict[str, Any]:
    training = sum(1 for record in records if record.split == "training")
    validation = sum(1 for record in records if record.split == "validation")
    return {"all_records": {"total": len(records), "training": training, "validation": validation}}


def reaction_records_content_hash(records: Sequence[TrainingReactionRecord]) -> str:
    signatures = sorted(reaction_record_signature(record) for record in records)
    return hash_json(signatures)


def reaction_record_signature(record: TrainingReactionRecord) -> tuple[Any, ...]:
    return (
        tuple(sorted(str(reactant) for reactant in record.reactants)),
        str(record.product),
        _condition_identity(record),
    )


def _condition_identity(record: TrainingReactionRecord) -> tuple[str, ...]:
    condition_smiles = tuple(str(value) for value in record.condition_slot_smiles)
    if condition_smiles:
        return condition_smiles
    return (record.condition_slot,) if record.condition_slot is not None else ()
