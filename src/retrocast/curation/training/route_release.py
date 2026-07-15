from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from retrocast._version import __version__
from retrocast.curation.training.records import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    RawRouteSource,
    TrainingRouteAdaptation,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
    TrainingSetBuildResult,
)
from retrocast.exceptions import AdapterError, TrainingReleaseError
from retrocast.hashing import hash_file, hash_json
from retrocast.io import ContentType, create_manifest, load_json_artifact, save_jsonl_gz
from retrocast.io.cache import json_type_cache, local_cache
from retrocast.models.route import Route

TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"


@local_cache(
    namespace="paroutes-adapted-routes",
    key=lambda source_path, *, dataset, **_: {
        "retrocast_version": __version__,
        "function": "adapt_training_routes",
        "source_sha256": hash_file(source_path),
        "dataset": dataset,
    },
    codec=json_type_cache(TrainingRouteAdaptation),
)
def adapt_training_routes(
    source_path: Path,
    *,
    dataset: str,
    show_progress: bool = True,
) -> TrainingRouteAdaptation:
    del show_progress  # Native adaptation owns the work; the cache remains at this boundary.
    from retrocast import native

    try:
        payload = native.adapt_training_routes(load_json_artifact(source_path), dataset)
    except native.NativeTrainingError as error:
        code = str(error.payload.get("code", "workflow.training_release_error"))
        message = str(error.payload.get("message", error))
        context = error.payload.get("context") if isinstance(error.payload.get("context"), dict) else {}
        if code.startswith("adapter."):
            raise AdapterError(message, code=code, context=context) from error
        raise TrainingReleaseError(message, code=code, context=context) from error
    return TrainingRouteAdaptation(
        routes=[
            AdaptedTrainingRoute(
                route=Route.model_validate(route["route"]),
                source=RawRouteSource.model_validate(route["source"]),
            )
            for route in payload["routes"]
        ],
        stats=AdaptationStatistics.model_validate(payload["stats"]),
    )


class TrainingRouteReleaseBuilder:
    def __init__(
        self,
        *,
        all_routes: Sequence[AdaptedTrainingRoute],
        all_adaptation: AdaptationStatistics,
        holdout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
        holdout_adaptation: Mapping[str, AdaptationStatistics],
        config: TrainingSetBuildConfig,
    ) -> None:
        self.all_routes = all_routes
        self.all_adaptation = all_adaptation
        self.holdout_routes = holdout_routes
        self.holdout_adaptation = holdout_adaptation
        self.config = config
        self._started = False

    def build(self) -> TrainingSetBuildResult:
        if self._started:
            raise RuntimeError("TrainingRouteReleaseBuilder instances are single-use")
        self._started = True

        from retrocast import native

        try:
            payload = native.build_training_route_release(
                self.all_routes,
                self.all_adaptation,
                self.holdout_routes,
                self.holdout_adaptation,
                self.config,
            )
        except native.NativeTrainingError as error:
            raise TrainingReleaseError(
                str(error.payload.get("message", error)),
                code=str(error.payload.get("code", "workflow.training_release_error")),
                context=error.payload.get("context") if isinstance(error.payload.get("context"), dict) else {},
            ) from error
        return TrainingSetBuildResult(
            release_name=payload["release_name"],
            records=[TrainingRouteRecord.model_validate(record) for record in payload["records"]],
            summary=payload["summary"],
        )


def write_training_release(
    *,
    result: TrainingSetBuildResult,
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
    config: TrainingSetBuildConfig,
) -> None:
    release_dir = output_dir / result.release_name
    all_path = release_dir / "all.jsonl.gz"
    training_path = release_dir / "training.jsonl.gz"
    validation_path = release_dir / "validation.jsonl.gz"
    manifest_path = release_dir / "manifest.json"
    training = [record for record in result.records if record.split == "training"]
    validation = [record for record in result.records if record.split == "validation"]
    save_jsonl_gz(result.records, all_path)
    save_jsonl_gz(training, training_path)
    save_jsonl_gz(validation, validation_path)
    manifest = create_manifest(
        action=TRAINING_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            ("all", all_path, result.records, ContentType.UNKNOWN, route_records_content_hash(result.records)),
            ("training", training_path, training, ContentType.UNKNOWN, route_records_content_hash(training)),
            ("validation", validation_path, validation, ContentType.UNKNOWN, route_records_content_hash(validation)),
        ],
        root_dir=source_root,
        parameters=config.to_manifest_dict(),
        statistics=result.summary.get("output", {}),
        summary=result.summary,
        release_name=result.release_name,
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def summarize_records(records: Sequence[TrainingRouteRecord]) -> dict[str, Any]:
    training = sum(1 for record in records if record.split == "training")
    validation = sum(1 for record in records if record.split == "validation")
    depths = Counter(record.route.depth() for record in records)
    return {
        "all_records": {"total": len(records), "training": training, "validation": validation},
        "by_depth": {str(depth): depths[depth] for depth in sorted(depths)},
    }


def route_records_content_hash(records: Sequence[TrainingRouteRecord]) -> str:
    signatures = sorted(record.route.signature() for record in records)
    return hash_json(signatures)
