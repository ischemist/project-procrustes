from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from retrocast.adapters.base import Adapter, AdaptMode
from retrocast.adapters.registry import ADAPTER_TYPES
from retrocast.metrics.constraints import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker
from retrocast.models.analysis import AnalysisReport
from retrocast.models.candidates import Candidate
from retrocast.models.evaluation import AcceptableRouteMatch, Evaluation
from retrocast.models.provenance import Manifest, VerificationReport
from retrocast.models.route import InChIKeyLevel, Route
from retrocast.models.task import Target, Task
from retrocast.utils.timing import ExecutionStats
from retrocast.workflow.collect import CollectedCandidates, NativeCollectedCandidates


class NativeDatasetError(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.payload = dict(payload)


class NativeTrainingError(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.payload = dict(payload)


class NativeEvaluation(Evaluation):
    """Python evaluation view retaining Rust ownership until mutable data is exposed."""

    _native_handle: Any | None = PrivateAttr(default=None)
    _materialized: bool = PrivateAttr(default=True)

    @classmethod
    def from_native_handle(cls, handle: Any) -> NativeEvaluation:
        """Create an unevaluated Python view without serializing the Rust graph."""
        evaluation = cls.model_construct()
        evaluation._native_handle = handle
        evaluation._materialized = False
        return evaluation

    def native_handle(self) -> Any | None:
        """Return the untouched handle for internal pipeline handoff."""
        return self._native_handle

    def materialize(self) -> Evaluation:
        """Expose the established Python DTO and end native fast-path ownership."""
        self._ensure_materialized()
        return self

    def _ensure_materialized(self) -> None:
        if self._materialized:
            return
        handle = self._native_handle
        if handle is None:
            raise RuntimeError("native evaluation lost its Rust value before materialization")
        materialized = Evaluation.model_validate_json(handle.json())
        object.__setattr__(self, "__dict__", materialized.__dict__)
        object.__setattr__(self, "__pydantic_extra__", materialized.__pydantic_extra__)
        object.__setattr__(self, "__pydantic_fields_set__", materialized.__pydantic_fields_set__)
        self._native_handle = None
        self._materialized = True

    def __getattribute__(self, name: str) -> Any:
        if name in type(self).model_fields:
            self._ensure_materialized()
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields:
            self._ensure_materialized()
        super().__setattr__(name, value)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._ensure_materialized()
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        self._ensure_materialized()
        return super().model_dump_json(*args, **kwargs)

    def model_copy(self, *args: Any, **kwargs: Any) -> NativeEvaluation:
        self._ensure_materialized()
        copied = super().model_copy(*args, **kwargs)
        copied._native_handle = None
        copied._materialized = True
        return copied

    def __eq__(self, other: object) -> bool:
        self._ensure_materialized()
        if isinstance(other, NativeEvaluation):
            other._ensure_materialized()
        if not isinstance(other, Evaluation):
            return False
        return self.__dict__ == other.__dict__ and (self.__pydantic_extra__ or {}) == (other.__pydantic_extra__ or {})

    def __repr__(self) -> str:
        self._ensure_materialized()
        return super().__repr__()


_native: Any | None
try:
    from retrocast import _native as native_extension
except ImportError:  # pragma: no cover - source checkouts before `maturin develop`
    _native = None
else:
    _native = native_extension


def available() -> bool:
    return _native is not None


def require_native() -> None:
    if _native is None:
        raise RuntimeError("RetroCast native engine is unavailable; install a platform wheel or run `maturin develop`.")


def _adapter_slug(adapter: Adapter) -> str | None:
    adapter_type = type(adapter)
    for slug, factory in ADAPTER_TYPES.items():
        if type(factory()) is adapter_type:
            if slug == "askcos" and getattr(adapter, "use_full_graph", False):
                return "askcosfullgraph"
            return slug
    return None


def adapt_route(
    raw_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
) -> Route | None:
    adapter_slug = _adapter_slug(adapter)
    if adapter_slug is None or not available():
        return None
    assert _native is not None
    try:
        payload = _native.adapt_route_json(
            _json(raw_payload),
            adapter_slug,
            mode=mode,
            target_json=target.model_dump_json(exclude_none=True) if target is not None else None,
        )
    except RuntimeError:
        return None
    return Route.model_validate_json(payload)


def adapt(
    raw_payload: Any,
    adapter: Adapter,
    *,
    mode: AdaptMode = "strict",
    target: Target | None = None,
    source_key: str | None = None,
    max_candidates: int | None = None,
    workers: int = 1,
) -> list[Candidate]:
    require_native()
    adapter_slug = _adapter_slug(adapter)
    if adapter_slug is None:
        raise NotImplementedError("the native engine cannot execute a custom Python adapter")
    assert _native is not None
    payload = _native.adapt_candidates_json(
        _json(raw_payload),
        adapter_slug,
        mode=mode,
        target_json=target.model_dump_json(exclude_none=True) if target is not None else None,
        source_key=source_key,
        max_candidates=max_candidates,
        workers=workers,
    )
    return [Candidate.model_validate(candidate) for candidate in json.loads(payload)]


def ingest(
    raw_payload: Any,
    adapter: Adapter,
    task: Task,
    *,
    mode: AdaptMode = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> CollectedCandidates:
    require_native()
    assert _native is not None
    adapter_slug = _adapter_slug(adapter)
    if adapter_slug is None:
        raise NotImplementedError("the native engine cannot execute a custom Python adapter")
    handle = _native.ingest_native(
        _json(raw_payload),
        adapter_slug,
        task.model_dump_json(exclude_none=True),
        mode=mode,
        max_candidates=max_candidates,
        workers=workers,
    )
    return NativeCollectedCandidates(handle)


def ingest_file(
    raw_path: Path,
    adapter: Adapter,
    task_path: Path,
    *,
    mode: AdaptMode = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> CollectedCandidates:
    """Ingest an artifact without constructing its raw graph in Python."""
    require_native()
    assert _native is not None
    adapter_slug = _adapter_slug(adapter)
    if adapter_slug is None:
        raise NotImplementedError("the native engine cannot execute a custom Python adapter")
    handle = _native.ingest_file_native(
        raw_path,
        adapter_slug,
        task_path,
        mode=mode,
        max_candidates=max_candidates,
        workers=workers,
    )
    return NativeCollectedCandidates(handle)


def load_predictions_file(path: Path) -> CollectedCandidates:
    require_native()
    assert _native is not None
    return NativeCollectedCandidates(_native.load_predictions_native(path))


def load_evaluation_file(path: Path) -> Evaluation:
    require_native()
    assert _native is not None
    return NativeEvaluation.from_native_handle(_native.load_evaluation_native(path))


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    constraint_checkers: Sequence[object],
    acceptable_match_level: InChIKeyLevel,
    acceptable_route_match: AcceptableRouteMatch,
    execution_stats: ExecutionStats | None,
    workers: int,
) -> Evaluation:
    require_native()
    assert _native is not None
    stocks: dict[str, set[str]] = {}
    supported = (StockTerminationChecker, RequiredLeavesChecker, RouteDepthChecker)
    if any(not isinstance(checker, supported) for checker in constraint_checkers):
        raise NotImplementedError("the native scorer received a custom task constraint checker")
    for checker in constraint_checkers:
        if isinstance(checker, StockTerminationChecker):
            stocks.update(checker.stock_keys_by_name)
    stats_json = (
        _json({"wall_time": execution_stats.wall_time, "cpu_time": execution_stats.cpu_time})
        if execution_stats is not None
        else None
    )
    predictions_handle = predictions.native_handle() if isinstance(predictions, NativeCollectedCandidates) else None
    if predictions_handle is not None:
        handle = _native.score_native(
            predictions_handle,
            task.model_dump_json(exclude_none=True),
            _json({name: sorted(values) for name, values in stocks.items()}),
            match_level=acceptable_match_level.value,
            acceptable_route_match=acceptable_route_match.value,
            execution_stats_json=stats_json,
            workers=workers,
        )
        return NativeEvaluation.from_native_handle(handle)

    predictions_json = _json(
        {
            target_id: [candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates]
            for target_id, candidates in predictions.items()
        }
    )
    payload = _native.score_json(
        predictions_json,
        task.model_dump_json(exclude_none=True),
        _json({name: sorted(values) for name, values in stocks.items()}),
        match_level=acceptable_match_level.value,
        acceptable_route_match=acceptable_route_match.value,
        execution_stats_json=stats_json,
        workers=workers,
    )
    return Evaluation.model_validate_json(payload)


def score_project_files(
    predictions_path: Path,
    task_path: Path,
    stocks_dir: Path,
    *,
    execution_stats_path: Path | None,
    acceptable_match_level: InChIKeyLevel,
    acceptable_route_match: AcceptableRouteMatch,
    workers: int = 1,
) -> tuple[Evaluation, str, list[Path]]:
    """Score project artifacts while every corpus-sized value remains in Rust."""
    require_native()
    assert _native is not None
    handle, label, stock_paths = _native.score_project_native(
        predictions_path,
        task_path,
        stocks_dir,
        execution_stats_path=execution_stats_path,
        match_level=acceptable_match_level.value,
        acceptable_route_match=acceptable_route_match.value,
        workers=workers,
    )
    return NativeEvaluation.from_native_handle(handle), label, [Path(path) for path in stock_paths]


def analyze(
    evaluation: Evaluation,
    *,
    ks: Sequence[int],
    prefix_depths: Sequence[int],
    n_boot: int,
    seed: int,
    workers: int,
) -> AnalysisReport:
    require_native()
    assert _native is not None
    evaluation_handle = evaluation.native_handle() if isinstance(evaluation, NativeEvaluation) else None
    if evaluation_handle is not None:
        payload = _native.analyze_native(
            evaluation_handle,
            list(ks),
            list(prefix_depths),
            n_boot=n_boot,
            seed=seed,
            workers=workers,
        )
    else:
        payload = _native.analyze_json(
            evaluation.model_dump_json(exclude_none=True),
            list(ks),
            list(prefix_depths),
            n_boot=n_boot,
            seed=seed,
            workers=workers,
        )
    return AnalysisReport.model_validate_json(payload)


def analyze_file(
    evaluation_path: Path,
    *,
    ks: Sequence[int],
    prefix_depths: Sequence[int],
    execution_stats_path: Path | None,
    n_boot: int,
    seed: int = 42,
    workers: int = 1,
) -> AnalysisReport:
    """Analyze an evaluation artifact without materializing it in Python."""
    require_native()
    assert _native is not None
    payload = _native.analyze_file_json(
        evaluation_path,
        list(ks),
        list(prefix_depths),
        execution_stats_path=execution_stats_path,
        n_boot=n_boot,
        seed=seed,
        workers=workers,
    )
    return AnalysisReport.model_validate_json(payload)


def run_pipeline(
    raw_path: Path,
    benchmark_path: Path,
    stock_path: Path,
    output_dir: Path,
    *,
    stock_name: str | None,
    execution_stats_path: Path | None,
    adapter: str,
    workers: int,
    mode: AdaptMode,
    max_candidates: int | None,
    match_level: InChIKeyLevel,
    acceptable_route_match: AcceptableRouteMatch,
    ks: Sequence[int],
    prefix_depths: Sequence[int],
    n_boot: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the file pipeline without transferring corpus-sized values to Python."""
    require_native()
    assert _native is not None
    return json.loads(
        _native.run_pipeline_json(
            raw_path,
            benchmark_path,
            stock_path,
            output_dir,
            stock_name=stock_name,
            execution_stats_path=execution_stats_path,
            adapter=adapter,
            workers=workers,
            mode=mode,
            max_candidates=max_candidates,
            match_level=match_level.value,
            acceptable_route_match=acceptable_route_match.value,
            ks=list(ks),
            prefix_depths=list(prefix_depths),
            n_boot=n_boot,
            seed=seed,
        )
    )


def verify_manifest(
    manifest_path: str,
    root_dir: str,
    *,
    deep: bool,
    output_only: bool,
    lenient: bool,
) -> VerificationReport:
    require_native()
    assert _native is not None
    payload = _native.verify_manifest_json(
        manifest_path,
        root_dir,
        deep=deep,
        output_only=output_only,
        lenient=lenient,
    )
    return VerificationReport.model_validate_json(payload)


def create_manifest(request: Mapping[str, Any]) -> Manifest:
    require_native()
    assert _native is not None
    return Manifest.model_validate_json(_native.create_manifest_json(_json(request)))


def hash_file(path: str) -> str:
    require_native()
    assert _native is not None
    return _native.hash_file(path)


def hash_json(value: Any) -> str:
    require_native()
    assert _native is not None
    return _native.hash_json(_json(value))


def write_json_gz(path: str, value: Any) -> None:
    require_native()
    assert _native is not None
    _native.write_json_gz_json(path, _json(value))


def write_jsonl_gz(path: str, rows: Sequence[Any]) -> int:
    require_native()
    assert _native is not None
    return _native.write_jsonl_gz_json(path, _json(rows))


def write_lines_gz(path: str, lines: Sequence[str]) -> int:
    require_native()
    assert _native is not None
    return _native.write_lines_gz(path, list(lines))


def write_csv_gz(path: str, rows: Sequence[Sequence[str]]) -> int:
    require_native()
    assert _native is not None
    return _native.write_csv_gz(path, [list(row) for row in rows])


def read_json(path: str) -> Any:
    require_native()
    assert _native is not None
    return json.loads(_native.read_json_json(path))


def read_jsonl(path: str, *, skip_empty: bool = True) -> list[Any]:
    require_native()
    assert _native is not None
    return json.loads(_native.read_jsonl_json(path, skip_empty=skip_empty))


def read_lines_gz(path: str) -> list[str]:
    require_native()
    assert _native is not None
    return _native.read_lines_gz(path)


def _dataset_call(operation: Any, *args: Any) -> Any:
    try:
        return operation(*args)
    except RuntimeError as error:
        message = str(error)
        prefix = "__retrocast_dataset__"
        if message.startswith(prefix):
            raise NativeDatasetError(json.loads(message.removeprefix(prefix))) from error
        raise


def dataset_build_url(base_url: str, segments: Sequence[str]) -> str:
    require_native()
    assert _native is not None
    return _native.dataset_build_url(base_url, list(segments))


def dataset_load_json_url(url: str) -> Any:
    require_native()
    assert _native is not None
    return json.loads(_dataset_call(_native.dataset_load_json_url_json, url))


def dataset_download_url_to_path(url: str, destination: str) -> None:
    require_native()
    assert _native is not None
    _dataset_call(_native.dataset_download_url_to_path, url, destination)


def dataset_load_sha256sums(path: str) -> dict[str, str]:
    require_native()
    assert _native is not None
    return dict(json.loads(_dataset_call(_native.dataset_load_sha256sums_json, path)))


def dataset_download_training_set(request: Mapping[str, Any]) -> str:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_download_training_set_json, _json(request))


def dataset_download_training_data(request: Mapping[str, Any]) -> list[str]:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_download_training_data_json, _json(request))


def dataset_download_hosted_data(request: Mapping[str, Any]) -> list[str]:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_download_hosted_data_json, _json(request))


def dataset_download_hosted_file(request: Mapping[str, Any]) -> str:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_download_hosted_file, _json(request))


def dataset_validate_training_request(dataset: str, artifact: str, split: str, format: str) -> None:
    require_native()
    assert _native is not None
    _dataset_call(_native.dataset_validate_training_request, dataset, artifact, split, format)


def dataset_resolve_release(dataset: str, release: str, base_url: str) -> str:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_resolve_release, dataset, release, base_url)


def dataset_training_filename(artifact: str, split: str, format: str) -> str:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_training_filename, artifact, split, format)


def dataset_training_root(
    dataset: str,
    release: str,
    cache_dir: str | None,
    output_dir: str | None,
) -> str:
    require_native()
    assert _native is not None
    return _native.dataset_training_root(dataset, release, cache_dir, output_dir)


def dataset_hosted_root(cache_dir: str | None, output_dir: str | None) -> str:
    require_native()
    assert _native is not None
    return _native.dataset_hosted_root(cache_dir, output_dir)


def dataset_training_file_matches(
    key: str,
    *,
    artifact: str | None,
    split: str | None,
    format: str | None,
    omit: Sequence[str],
) -> bool:
    require_native()
    assert _native is not None
    return _native.dataset_training_file_matches(
        key,
        artifact=artifact,
        split=split,
        format=format,
        omit=list(omit),
    )


def dataset_resolve_expected(path: str, url: str, key: str) -> str:
    require_native()
    assert _native is not None
    return _dataset_call(_native.dataset_resolve_expected, path, url, key)


def find_route_embeddings(
    query: Route,
    container: Route,
    *,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
) -> list[dict[str, Any]]:
    require_native()
    assert _native is not None
    payload = _native.find_route_embeddings_json(
        query.model_dump_json(exclude_none=True),
        container.model_dump_json(exclude_none=True),
        match_level=match_level.value,
        allow_leaf_extension=allow_leaf_extension,
    )
    return json.loads(payload)


def route_embeds_at(
    query: Any,
    container: Any,
    *,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
) -> dict[str, Any] | None:
    require_native()
    assert _native is not None
    query_route = query.route
    query_path = query.path
    container_route = container.route
    container_path = container.path
    payload = _native.route_embeds_at_json(
        query_route.model_dump_json(exclude_none=True),
        query_path.id(),
        container_route.model_dump_json(exclude_none=True),
        container_path.id(),
        match_level=match_level.value,
        allow_leaf_extension=allow_leaf_extension,
    )
    return json.loads(payload)


def subtree_reaction_count(molecule: Any) -> int:
    require_native()
    assert _native is not None
    return _native.subtree_reaction_count_json(
        molecule.route.model_dump_json(exclude_none=True),
        molecule.path.id(),
    )


def excise_reactions(route: Route, excluded: set[str]) -> list[Route]:
    require_native()
    assert _native is not None
    payload = _native.excise_reactions_json(route.model_dump_json(exclude_none=True), sorted(excluded))
    return [Route.model_validate(value) for value in json.loads(payload)]


def deduplicate_routes(routes: Sequence[Route]) -> list[Route]:
    require_native()
    assert _native is not None
    payload = _native.deduplicate_routes_json(
        _json([route.model_dump(mode="json", exclude_none=True) for route in routes])
    )
    return [Route.model_validate(value) for value in json.loads(payload)]


def filter_by_route_type(task: Task, route_type: str) -> list[Target]:
    require_native()
    assert _native is not None
    payload = _native.filter_by_route_type_json(task.model_dump_json(exclude_none=True), route_type)
    return [Target.model_validate(value) for value in json.loads(payload)]


def clean_and_prioritize_pools(
    primary: Sequence[Target], secondary: Sequence[Target]
) -> tuple[list[Target], list[Target]]:
    require_native()
    assert _native is not None
    payload = _native.clean_and_prioritize_pools_json(
        _json([target.model_dump(mode="json", exclude_none=True) for target in primary]),
        _json([target.model_dump(mode="json", exclude_none=True) for target in secondary]),
    )
    primary_values, secondary_values = json.loads(payload)
    return (
        [Target.model_validate(value) for value in primary_values],
        [Target.model_validate(value) for value in secondary_values],
    )


def generate_pruned_routes(route: Route, stock: set[str]) -> list[Route]:
    require_native()
    assert _native is not None
    payload = _native.generate_pruned_routes_json(route.model_dump_json(exclude_none=True), sorted(stock))
    return [Route.model_validate(value) for value in json.loads(payload)]


def route_is_convergent(route: Route) -> bool:
    require_native()
    assert _native is not None
    return bool(_native.route_is_convergent_json(route.model_dump_json(exclude_none=True)))


def sample_indices(population_size: int, sample_size: int, seed: int) -> list[int]:
    require_native()
    assert _native is not None
    return _native.sample_indices(population_size, sample_size, str(int(seed)))


def sample_stratified_priority_indices(
    grouped_pool_sizes: Sequence[Sequence[int]], target_counts: Sequence[int], seed: int
) -> list[list[tuple[int, int]]]:
    require_native()
    assert _native is not None
    return _native.sample_stratified_priority_indices(
        [list(pool_sizes) for pool_sizes in grouped_pool_sizes], list(target_counts), str(int(seed))
    )


def training_validation_indices(routes: Sequence[Route], validation_fraction: float, seed: int) -> list[int]:
    require_native()
    assert _native is not None
    return _native.training_validation_indices_json(
        _json([route.model_dump(mode="json", exclude_none=True) for route in routes]),
        validation_fraction,
        str(int(seed)),
    )


def _training_call(operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except RuntimeError as error:
        message = str(error)
        prefix = "__retrocast_training__"
        if message.startswith(prefix):
            raise NativeTrainingError(json.loads(message.removeprefix(prefix))) from error
        raise


def build_test_route_records(dataset: str, routes: Sequence[Any], route_prefix: str) -> list[dict[str, Any]]:
    require_native()
    assert _native is not None
    payload = _training_call(
        _native.build_test_route_records_json,
        dataset,
        _json([{"route": route.route, "source": route.source} for route in routes]),
        route_prefix=route_prefix,
    )
    return json.loads(payload)


def adapt_training_routes(raw: Any, dataset: str) -> dict[str, Any]:
    require_native()
    assert _native is not None
    payload = _training_call(_native.adapt_training_routes_json, _json(raw), dataset)
    return json.loads(payload)


def build_test_reaction_records(dataset: str, route_records: Sequence[Any], route_prefix: str) -> list[dict[str, Any]]:
    require_native()
    assert _native is not None
    payload = _training_call(
        _native.build_test_reaction_records_json,
        dataset,
        _json([record.model_dump(mode="json", exclude_none=True) for record in route_records]),
        route_prefix=route_prefix,
    )
    return json.loads(payload)


def build_training_reaction_release(route_records: Sequence[Any], config: Any) -> dict[str, Any]:
    require_native()
    assert _native is not None
    payload = _training_call(
        _native.build_training_reaction_release_json,
        _json([record.model_dump(mode="json", exclude_none=True) for record in route_records]),
        _json(
            {
                "holdout_mode": config.holdout_mode,
                "route_prefix": config.route_prefix,
            }
        ),
    )
    return json.loads(payload)


def build_training_route_release(
    all_routes: Sequence[Any],
    all_adaptation: Any,
    holdout_routes: Mapping[str, Sequence[Any]],
    holdout_adaptation: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    require_native()
    assert _native is not None
    payload = _training_call(
        _native.build_training_route_release_json,
        _json([{"route": route.route, "source": route.source} for route in all_routes]),
        _json(all_adaptation.to_manifest_dict()),
        _json(
            {
                name: [{"route": route.route, "source": route.source} for route in routes]
                for name, routes in holdout_routes.items()
            }
        ),
        _json({name: stats.to_manifest_dict() for name, stats in holdout_adaptation.items()}),
        _json(
            {
                "holdout_mode": config.holdout_mode,
                "val_fraction": config.val_fraction,
                "seed": config.seed,
                "route_prefix": config.route_prefix,
            }
        ),
    )
    return json.loads(payload)


def audit_route_release(
    release_name: str, all_records: Sequence[Any], training: Sequence[Any], validation: Sequence[Any]
) -> None:
    require_native()
    assert _native is not None
    _training_call(
        _native.audit_route_release_json,
        release_name,
        _json(list(all_records)),
        _json(list(training)),
        _json(list(validation)),
    )


def audit_single_step_release(
    release_name: str,
    all_records: Sequence[Any],
    training: Sequence[Any],
    validation: Sequence[Any],
    all_rsmi_count: int,
    training_rsmi_count: int,
    validation_rsmi_count: int,
    parent_route_ids: set[str],
) -> dict[str, Any]:
    require_native()
    assert _native is not None
    payload = _training_call(
        _native.audit_single_step_release_json,
        release_name,
        _json(list(all_records)),
        _json(list(training)),
        _json(list(validation)),
        all_rsmi_count,
        training_rsmi_count,
        validation_rsmi_count,
        sorted(parent_route_ids),
    )
    return json.loads(payload)


def build_route_embedding_audit(
    release_name: str,
    container_records: Sequence[Any],
    queries_by_source: Mapping[str, Mapping[str, Route]],
    *,
    match_level: InChIKeyLevel,
    allow_leaf_extension: bool,
    include_partial: bool,
    partial_min_reactions: int,
    exclude_query_containers: bool,
) -> dict[str, Any]:
    require_native()
    assert _native is not None
    payload = _native.build_route_embedding_audit_json(
        release_name,
        _json(list(container_records)),
        _json(
            [
                {
                    "source": source,
                    "queries": [{"id": query_id, "route": route} for query_id, route in queries.items()],
                }
                for source, queries in sorted(queries_by_source.items())
            ]
        ),
        _json(
            {
                "match_level": match_level.value,
                "allow_leaf_extension": allow_leaf_extension,
                "include_partial": include_partial,
                "partial_min_reactions": partial_min_reactions,
                "exclude_query_containers": exclude_query_containers,
            }
        ),
    )
    return json.loads(payload)


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
